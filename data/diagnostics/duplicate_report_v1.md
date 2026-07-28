# Duplicate Grouping Report v1

- Input rows: **17,880**
- Exact duplicate copies removed: **2,073**
- Rows kept after exact deduplication: **15,807**
- Final groups: **14,673**
- Groups containing more than one row: **651**
- Exact duplicate groups: **726**
- Rows inside exact duplicate groups: **2,799**
- Near-duplicate pairs found: **4,998** from 61,635 checked pairs
- Largest final group: **82 rows**
- Groups with both labels: **0**

## Rules used

- Exact duplicate: same combined text after lowercasing and collapsing whitespace.
- Near duplicate: same cleaned title and word-set Jaccard similarity >= 0.90.
- The first record in source order is kept for each exact text.
- Different near-duplicate texts are retained and assigned the same group ID.
- Labels of retained records are not changed.

## Label conflicts

No groups contain conflicting labels.

## Limitation

Near-duplicate checking only compares advertisements with the same cleaned title. Similar templates with different titles may be missed.
