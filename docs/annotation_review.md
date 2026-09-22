# Reviewing annotations

Use the local review page to inspect and correct any active annotation CSV:

```bash
.venv/bin/annotation-review-server
```

Open <http://127.0.0.1:8765>. The file menu discovers assembled/final `.csv`
files below `data/annotations`; intermediate `*_chunks/` files are deliberately
excluded. Edit labels or evidence across any rows, then click the sticky
**Save all changes** button. The complete document is written in one atomic
operation, so a
partial write cannot replace the original file. The page does not hard-code the
taxonomy: it displays whatever columns exist in the selected CSV and suggests
the primary labels already present in that file.

For another annotation directory or port:

```bash
.venv/bin/annotation-review-server --annotations path/to/annotations --port 8766
```

Keep the server bound to its default loopback host unless access from another
machine is explicitly needed. Manually changed rows are marked with
`annotator=manual`.
