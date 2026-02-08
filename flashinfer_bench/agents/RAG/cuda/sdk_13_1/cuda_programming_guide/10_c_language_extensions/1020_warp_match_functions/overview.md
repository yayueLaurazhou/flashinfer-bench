# 10.20. Warp Match Functions

## 10.20. Warp Match Functions[](#warp-match-functions "Permalink to this headline")

`__match_any_sync` and `__match_all_sync` perform a broadcast-and-compare operation of a variable between threads within a [warp](#simt-architecture).

Supported by devices of compute capability 7.x or higher.