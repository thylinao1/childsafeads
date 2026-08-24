# Upstream code not stored in this repo

`mirror/` is a clone of a public repository that belongs to someone else. It
is 370M and 225 files. Re-uploading it here would duplicate
code that already has a home, and git cannot track a nested repository's files
anyway, so it is excluded and pinned below instead.

Both facts were verified on 2026-08-13: the remote answered `git ls-remote`,
and the commit below is what was checked out locally.

| | |
|---|---|
| path | `mirror/` |
| remote | https://github.com/aiejvn/childsafeads_emnllp |
| commit | `05bacc6b8d82a4ea9a4592b28947c405a60abc08` |

## Restore it

```bash
git clone https://github.com/aiejvn/childsafeads_emnllp mirror
cd mirror && git checkout 05bacc6b8d82a4ea9a4592b28947c405a60abc08
```

This repository does not keep a copy of that commit; if the upstream repository is
removed, the pinned code cannot be recovered from here.
