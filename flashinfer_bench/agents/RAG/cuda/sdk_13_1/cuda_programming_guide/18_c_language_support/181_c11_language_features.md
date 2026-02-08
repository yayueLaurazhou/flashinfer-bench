# 18.1. C++11 Language Features

## 18.1. C++11 Language Features[](#c-11-language-features "Permalink to this headline")

The following table lists new language features that have been accepted into the C++11 standard. The “Proposal” column provides a link to the ISO C++ committee proposal that describes the feature, while the “Available in nvcc (device code)” column indicates the first version of nvcc that contains an implementation of this feature (if it has been implemented) for device code.

Table 23 C++11 Language Features[](#id483 "Permalink to this table")





| Language Feature | C++11 Proposal | Available in nvcc (device code) |
| --- | --- | --- |
| Rvalue references | [N2118](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html) | 7.0 |
| Rvalue references for `*this` | [N2439](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm) | 7.0 |
| Initialization of class objects by rvalues | [N1610](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html) | 7.0 |
| Non-static data member initializers | [N2756](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm) | 7.0 |
| Variadic templates | [N2242](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf) | 7.0 |
| Extending variadic template template parameters | [N2555](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf) | 7.0 |
| Initializer lists | [N2672](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm) | 7.0 |
| Static assertions | [N1720](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html) | 7.0 |
| `auto`-typed variables | [N1984](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf) | 7.0 |
| Multi-declarator `auto` | [N1737](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf) | 7.0 |
| Removal of auto as a storage-class specifier | [N2546](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm) | 7.0 |
| New function declarator syntax | [N2541](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm) | 7.0 |
| Lambda expressions | [N2927](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf) | 7.0 |
| Declared type of an expression | [N2343](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf) | 7.0 |
| Incomplete return types | [N3276](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf) | 7.0 |
| Right angle brackets | [N1757](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html) | 7.0 |
| Default template arguments for function templates | [DR226](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226) | 7.0 |
| Solving the SFINAE problem for expressions | [DR339](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html) | 7.0 |
| Alias templates | [N2258](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf) | 7.0 |
| Extern templates | [N1987](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm) | 7.0 |
| Null pointer constant | [N2431](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf) | 7.0 |
| Strongly-typed enums | [N2347](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf) | 7.0 |
| Forward declarations for enums | [N2764](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf) [DR1206](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206) | 7.0 |
| Standardized attribute syntax | [N2761](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf) | 7.0 |
| Generalized constant expressions | [N2235](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf) | 7.0 |
| Alignment support | [N2341](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf) | 7.0 |
| Conditionally-support behavior | [N1627](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf) | 7.0 |
| Changing undefined behavior into diagnosable errors | [N1727](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf) | 7.0 |
| Delegating constructors | [N1986](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf) | 7.0 |
| Inheriting constructors | [N2540](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm) | 7.0 |
| Explicit conversion operators | [N2437](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf) | 7.0 |
| New character types | [N2249](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html) | 7.0 |
| Unicode string literals | [N2442](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm) | 7.0 |
| Raw string literals | [N2442](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm) | 7.0 |
| Universal character names in literals | [N2170](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html) | 7.0 |
| User-defined literals | [N2765](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf) | 7.0 |
| Standard Layout Types | [N2342](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm) | 7.0 |
| Defaulted functions | [N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm) | 7.0 |
| Deleted functions | [N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm) | 7.0 |
| Extended friend declarations | [N1791](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf) | 7.0 |
| Extending `sizeof` | [N2253](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html) [DR850](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#850) | 7.0 |
| Inline namespaces | [N2535](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm) | 7.0 |
| Unrestricted unions | [N2544](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf) | 7.0 |
| Local and unnamed types as template arguments | [N2657](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm) | 7.0 |
| Range-based for | [N2930](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html) | 7.0 |
| Explicit virtual overrides | [N2928](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm) [N3206](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm) [N3272](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm) | 7.0 |
| Minimal support for garbage collection and reachability-based leak detection | [N2670](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm) | N/A (see [Restrictions](#language-restrictions)) |
| Allowing move constructors to throw [noexcept] | [N3050](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html) | 7.0 |
| Defining move special member functions | [N3053](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html) | 7.0 |
| **Concurrency** | | |
| Sequence points | [N2239](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html) |  |
| Atomic operations | [N2427](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html) |  |
| Strong Compare and Exchange | [N2748](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html) |  |
| Bidirectional Fences | [N2752](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm) |  |
| Memory model | [N2429](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm) |  |
| Data-dependency ordering: atomics and memory model | [N2664](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm) |  |
| Propagating exceptions | [N2179](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html) |  |
| Allow atomics use in signal handlers | [N2547](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm) |  |
| Thread-local storage | [N2659](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm) |  |
| Dynamic initialization and destruction with concurrency | [N2660](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm) |  |
| **C99 Features in C++11** | | |
| `__func__` predefined identifier | [N2340](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm) | 7.0 |
| C99 preprocessor | [N1653](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm) | 7.0 |
| `long long` | [N1811](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf) | 7.0 |
| Extended integral types | [N1988](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf) |  |