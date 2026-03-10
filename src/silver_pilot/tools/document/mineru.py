"""
模块名称：mineru
功能描述：MinerU 转Markdown 服务模块，封装了文件上传、提取结果查询、结果下载
         以及完整的 MD 转换工作流，提供统一的异常处理与日志记录。
"""

import io
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import requests
from pydantic import BaseModel, Field

# ========== 项目配置 ==========
from silver_pilot.config import config
from silver_pilot.utils import get_channel_logger

logger = get_channel_logger(log_dir=config.LOG_DIR / "mineru", channel_name="MinerU")

# ========== 常量 ==========

MINERU_BASE_URL = "https://mineru.net/api/v4"
DEFAULT_POLL_INTERVAL = 20  # 轮询间隔（秒）
DEFAULT_TIMEOUT = 600  # 最大等待时间（秒）
DEFAULT_MODEL_VERSION = "vlm"


# ========== 异常定义 ==========


class MinerUError(Exception):
    """MinerU 服务基础异常"""

    def __init__(self, message: str, code: int | None = None, trace_id: str | None = None):
        self.code = code
        self.trace_id = trace_id
        super().__init__(message)


class MinerUUploadError(MinerUError):
    """文件上传阶段异常"""


class MinerUExtractError(MinerUError):
    """提取结果阶段异常"""


class MinerUDownloadError(MinerUError):
    """结果下载阶段异常"""


class MinerUTimeoutError(MinerUError):
    """轮询超时异常"""


# ========== 数据模型 ==========


class MinerUFileRequest(BaseModel):
    """单个文件的上传参数"""

    name: str = Field(description="文件名称（含路径）")
    is_ocr: bool = Field(default=True, description="是否启用OCR")
    data_id: str = Field(default="", description="用户自定义文件标识")


class MinerUUploadResponse(BaseModel):
    """上传接口返回的数据"""

    batch_id: str = Field(description="批次ID")
    file_urls: list[str] = Field(description="预签名上传URL列表")


class MinerUExtractProgress(BaseModel):
    """文件提取进度"""

    extracted_pages: int = Field(default=0, description="已提取页数")
    total_pages: int = Field(default=0, description="总页数")
    start_time: str = Field(default="", description="开始时间")


class MinerUExtractResult(BaseModel):
    """单个文件的提取结果"""

    file_name: str = Field(description="文件名")
    state: str = Field(description="状态：done / running / failed")
    err_msg: str = Field(default="", description="错误信息")
    full_zip_url: str | None = Field(default=None, description="结果压缩包下载URL")
    extract_progress: MinerUExtractProgress | None = Field(
        default=None, description="提取进度（running时存在）"
    )

    @property
    def is_done(self) -> bool:
        return self.state == "done"

    @property
    def is_running(self) -> bool:
        return self.state == "running"

    @property
    def is_failed(self) -> bool:
        return self.state == "failed"


class MinerUExtractResponse(BaseModel):
    """提取结果查询返回的数据"""

    batch_id: str = Field(description="批次ID")
    extract_result: list[MinerUExtractResult] = Field(description="各文件提取结果列表")


# ========== 服务类 ==========


class MinerUService:
    """
    MinerU Markdown 转换服务。

    使用示例::

        service = MinerUService()
        results = service.process(
            file_paths=["path/to/document.pdf"],
            output_dir=Path("output"),
        )
    """

    def __init__(self, token: str | None = None, base_url: str = MINERU_BASE_URL):
        """
        初始化 MinerU 服务。

        Args:
            token: API Token，默认从配置中读取 MINERU_TOKEN
            base_url: API 基础 URL
        """
        self.token = token or config.MINERU_TOKEN
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    # ------------------------------------------------------------------
    # 1. 申请上传 URL
    # ------------------------------------------------------------------

    def apply_upload_urls(
        self,
        files: list[MinerUFileRequest],
        model_version: str = DEFAULT_MODEL_VERSION,
    ) -> MinerUUploadResponse:
        """
        向 MinerU 申请文件上传的预签名 URL。

        Args:
            files: 待上传文件列表
            model_version: 模型版本，默认 "vlm"

        Returns:
            MinerUUploadResponse: 包含 batch_id 和预签名 URL 列表

        Raises:
            MinerUUploadError: 申请失败时抛出
        """
        url = f"{self.base_url}/file-urls/batch"
        payload = {
            "files": [f.model_dump() for f in files],
            "model_version": model_version,
        }

        logger.info(f"申请上传URL，文件数量: {len(files)}")

        try:
            response = requests.post(url, headers=self._headers, json=payload, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"申请上传URL网络请求失败: {e}")
            raise MinerUUploadError(f"申请上传URL请求失败: {e}") from e

        result = response.json()
        code = result.get("code")
        trace_id = result.get("trace_id", "")

        if code != 0:
            msg = result.get("msg", "未知错误")
            logger.error(f"申请上传URL业务失败: code={code}, msg={msg}, trace_id={trace_id}")
            raise MinerUUploadError(f"申请上传URL失败: {msg}", code=code, trace_id=trace_id)

        data = MinerUUploadResponse(**result["data"])
        logger.info(f"申请上传URL成功，batch_id={data.batch_id}，URL数量={len(data.file_urls)}")
        return data

    # ------------------------------------------------------------------
    # 2. 上传文件
    # ------------------------------------------------------------------

    def upload_files(
        self,
        file_paths: list[str | Path],
        upload_urls: list[str],
    ) -> None:
        """
        将本地文件通过 PUT 上传到预签名 URL。

        Args:
            file_paths: 本地文件路径列表
            upload_urls: 对应的预签名 URL 列表

        Raises:
            MinerUUploadError: 上传失败时抛出
        """
        if len(file_paths) != len(upload_urls):
            raise MinerUUploadError(
                f"文件数量({len(file_paths)})与URL数量({len(upload_urls)})不匹配"
            )

        for file_path, url in zip(file_paths, upload_urls):
            path = Path(file_path)
            if not path.exists():
                raise MinerUUploadError(f"文件不存在: {path}")

            logger.info(f"正在上传文件: {path.name}")

            try:
                with open(path, "rb") as f:
                    resp = requests.put(url, data=f, timeout=300)
                    resp.raise_for_status()
            except requests.RequestException as e:
                logger.error(f"文件上传失败: {path.name}, 错误: {e}")
                raise MinerUUploadError(f"文件 {path.name} 上传失败: {e}") from e

            logger.info(f"文件上传成功: {path.name}")

    # ------------------------------------------------------------------
    # 3. 查询提取结果
    # ------------------------------------------------------------------

    def get_extract_results(self, batch_id: str) -> MinerUExtractResponse:
        """
        查询指定批次的提取结果。

        Args:
            batch_id: 批次ID

        Returns:
            MinerUExtractResponse: 批次内所有文件的提取状态与结果

        Raises:
            MinerUExtractError: 查询失败时抛出
        """
        url = f"{self.base_url}/extract-results/batch/{batch_id}"

        try:
            response = requests.get(url, headers=self._headers, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"查询提取结果网络请求失败: {e}")
            raise MinerUExtractError(f"查询提取结果请求失败: {e}") from e

        result = response.json()
        code = result.get("code")
        trace_id = result.get("trace_id", "")

        if code != 0:
            msg = result.get("msg", "未知错误")
            logger.error(f"查询提取结果业务失败: code={code}, msg={msg}, trace_id={trace_id}")
            raise MinerUExtractError(f"查询提取结果失败: {msg}", code=code, trace_id=trace_id)

        data = MinerUExtractResponse(**result["data"])
        return data

    # ------------------------------------------------------------------
    # 4. 轮询等待完成
    # ------------------------------------------------------------------

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> MinerUExtractResponse:
        """
        轮询等待批次内所有文件提取完成。

        Args:
            batch_id: 批次ID
            poll_interval: 轮询间隔秒数，默认20秒
            timeout: 最大等待秒数，默认600秒

        Returns:
            MinerUExtractResponse: 最终提取结果

        Raises:
            MinerUTimeoutError: 超时未完成
            MinerUExtractError: 有文件提取失败
        """
        logger.info(f"开始轮询提取结果，batch_id={batch_id}，超时={timeout}s")
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise MinerUTimeoutError(f"轮询超时（已等待 {int(elapsed)}s），batch_id={batch_id}")

            data = self.get_extract_results(batch_id)

            # 检查是否有失败的文件
            failed = [r for r in data.extract_result if r.is_failed]
            if failed:
                names = ", ".join(r.file_name for r in failed)
                errors = "; ".join(f"{r.file_name}: {r.err_msg}" for r in failed)
                logger.error(f"文件提取失败: {names}")
                raise MinerUExtractError(f"文件提取失败: {errors}")

            # 检查是否全部完成
            all_done = all(r.is_done for r in data.extract_result)
            if all_done:
                logger.info(f"所有文件提取完成，batch_id={batch_id}，耗时={int(elapsed)}s")
                return data

            # 打印进度
            running = [r for r in data.extract_result if r.is_running]
            for r in running:
                progress = r.extract_progress
                if progress:
                    logger.info(
                        f"  {r.file_name}: {progress.extracted_pages}/{progress.total_pages} 页"
                    )

            logger.info(f"等待 {poll_interval}s 后重新查询...")
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # 5. 下载并解压结果
    # ------------------------------------------------------------------

    def download_result(
        self,
        extract_result: MinerUExtractResult,
        output_dir: Path,
    ) -> list[Path]:
        """
        下载单个文件的提取结果 zip，仅提取其中的 .md 文件到指定目录。

        Args:
            extract_result: 已完成的提取结果
            output_dir: 输出目录（.md 文件将直接存放于此）

        Returns:
            list[Path]: 提取出的 .md 文件路径列表

        Raises:
            MinerUDownloadError: 下载或解压失败
        """
        if not extract_result.is_done or not extract_result.full_zip_url:
            raise MinerUDownloadError(f"文件 {extract_result.file_name} 尚未完成或无下载链接")

        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"正在下载结果: {extract_result.file_name}")

        try:
            resp = requests.get(extract_result.full_zip_url, timeout=300)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"下载结果失败: {extract_result.file_name}, 错误: {e}")
            raise MinerUDownloadError(f"下载 {extract_result.file_name} 结果失败: {e}") from e

        md_files: list[Path] = []
        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                # 解压到临时目录，再筛选 .md 文件
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zf.extractall(tmp_dir)
                    for md_path in Path(tmp_dir).rglob("*.md"):
                        dest = (
                            output_dir
                            / f"{Path(extract_result.file_name).with_suffix('')}_{md_path.stem}.md"
                        )
                        shutil.copy2(md_path, dest)
                        md_files.append(dest)
                        logger.info(f"已保存: {dest}")
        except zipfile.BadZipFile as e:
            logger.error(f"解压结果失败: {extract_result.file_name}, 错误: {e}")
            raise MinerUDownloadError(f"解压 {extract_result.file_name} 结果失败: {e}") from e

        if not md_files:
            logger.warning(f"压缩包中未找到 .md 文件: {extract_result.file_name}")

        return md_files

    def download_results(
        self,
        extract_response: MinerUExtractResponse,
        output_dir: Path,
    ) -> list[Path]:
        """
        批量下载所有已完成文件的提取结果，仅保留 .md 文件。

        Args:
            extract_response: 提取结果响应
            output_dir: 输出目录

        Returns:
            list[Path]: 所有提取出的 .md 文件路径列表
        """
        all_md_files: list[Path] = []
        for result in extract_response.extract_result:
            if result.is_done and result.full_zip_url:
                md_files = self.download_result(result, output_dir)
                all_md_files.extend(md_files)
        return all_md_files

    # ------------------------------------------------------------------
    # 6. 一站式完整流程
    # ------------------------------------------------------------------

    def process(
        self,
        file_paths: list[str | Path],
        output_dir: str | Path,
        *,
        is_ocr: bool = True,
        model_version: str = DEFAULT_MODEL_VERSION,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> list[Path]:
        """
        一站式 Markdown 转换流程：上传 → 轮询 → 下载。

        Args:
            file_paths: 本地文件路径列表或html文件路径列表
            output_dir: 结果输出目录
            is_ocr: 是否启用OCR，默认True
            model_version: 模型版本，默认 "vlm"，html文件请使用 "MinerU-HTML"
            poll_interval: 轮询间隔秒数
            timeout: 最大等待秒数

        Returns:
            list[Path]: 提取出的 .md 文件路径列表

        Raises:
            MinerUError: 任一阶段发生错误
        """
        output_path = Path(output_dir)
        logger.info(f"===== 开始 MD 转换流程，共 {len(file_paths)} 个文件 =====")

        # Step 1: 构建文件请求
        file_requests = [MinerUFileRequest(name=Path(fp).name, is_ocr=is_ocr) for fp in file_paths]

        # Step 2: 申请上传 URL
        upload_resp = self.apply_upload_urls(file_requests, model_version=model_version)
        logger.info(f"批次ID: {upload_resp.batch_id}")

        # Step 3: 上传文件
        self.upload_files(file_paths, upload_resp.file_urls)

        # Step 4: 轮询等待完成
        extract_resp = self.wait_for_completion(
            upload_resp.batch_id,
            poll_interval=poll_interval,
            timeout=timeout,
        )

        # Step 5: 下载结果（仅保留 .md 文件）
        md_files = self.download_results(extract_resp, output_path)

        logger.info(f"===== MD 转换流程完成，共获得 {len(md_files)} 个 .md 文件 =====")
        return md_files
