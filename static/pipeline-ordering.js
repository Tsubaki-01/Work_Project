(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    module.exports = factory();
    return;
  }
  root.PipelineOrdering = factory();
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function groupRank(groupId) {
    if (!groupId || typeof groupId !== "string") return 99;
    if (groupId.startsWith("0-")) return 0;
    if (groupId.startsWith("1-")) return 1;
    if (groupId.startsWith("2-")) return 2;
    return 99;
  }

  function normalizePipelineOrder(pipeline) {
    if (!Array.isArray(pipeline)) return [];

    pipeline.sort(function (a, b) {
      const ga = groupRank(a.group_id);
      const gb = groupRank(b.group_id);
      if (ga !== gb) return ga - gb;

      const sa = Number.isFinite(a.event_seq) ? a.event_seq : Number.MAX_SAFE_INTEGER;
      const sb = Number.isFinite(b.event_seq) ? b.event_seq : Number.MAX_SAFE_INTEGER;
      if (sa !== sb) return sa - sb;

      const ia = Number.isFinite(a._insert_idx) ? a._insert_idx : Number.MAX_SAFE_INTEGER;
      const ib = Number.isFinite(b._insert_idx) ? b._insert_idx : Number.MAX_SAFE_INTEGER;
      return ia - ib;
    });

    return pipeline;
  }

  return {
    groupRank,
    normalizePipelineOrder,
  };
});
