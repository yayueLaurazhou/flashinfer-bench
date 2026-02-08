# 18.7.2. Extended Lambda Restrictions

### 18.7.2. Extended Lambda Restrictions[](#extended-lambda-restrictions "Permalink to this headline")

The CUDA compiler will replace an extended lambda expression with an instance of a placeholder type defined in namespace scope, before invoking the host compiler. The template argument of the placeholder type requires taking the address of a function enclosing the original extended lambda expression. This is required for the correct execution of any `__global__` function template whose template argument involves the closure type of an extended lambda. The *enclosing function* is computed as follows.

By definition, the extended lambda is present within the immediate or nested block scope of a `__host__` or `__host__ __device__` function. If this function is not the `operator()` of a lambda expression, then it is considered the enclosing function for the extended lambda. Otherwise, the extended lambda is defined within the immediate or nested block scope of the `operator()` of one or more enclosing lambda expressions. If the outermost such lambda expression is defined in the immediate or nested block scope of a function `F`, then `F` is the computed enclosing function, else the enclosing function does not exist.

Example:

```
void foo(void) {
  // enclosing function for lam1 is "foo"
  auto lam1 = [] __device__ { };

  auto lam2 = [] {
     auto lam3 = [] {
        // enclosing function for lam4 is "foo"
        auto lam4 = [] __host__ __device__ { };
     };
  };
}

auto lam6 = [] {
  // enclosing function for lam7 does not exist
  auto lam7 = [] __host__ __device__ { };
};
```

Here are the restrictions on extended lambdas:

1. An extended lambda cannot be defined inside another extended lambda expression.

   Example:

   ```
   void foo(void) {
     auto lam1 = [] __host__ __device__  {
       // error: extended lambda defined within another extended lambda
       auto lam2 = [] __host__ __device__ { };
     };
   }
   ```
2. An extended lambda cannot be defined inside a generic lambda expression.

   Example:

   ```
   void foo(void) {
     auto lam1 = [] (auto) {
       // error: extended lambda defined within a generic lambda
       auto lam2 = [] __host__ __device__ { };
     };
   }
   ```
3. If an extended lambda is defined within the immediate or nested block scope of one or more nested lambda expression, the outermost such lambda expression must be defined inside the immediate or nested block scope of a function.

   Example:

   ```
   auto lam1 = []  {
     // error: outer enclosing lambda is not defined within a
     // non-lambda-operator() function.
     auto lam2 = [] __host__ __device__ { };
   };
   ```
4. The enclosing function for the extended lambda must be named and its address can be taken. If the enclosing function is a class member, then the following conditions must be satisfied:

   * All classes enclosing the member function must have a name.
   * The member function must not have private or protected access within its parent class.
   * All enclosing classes must not have private or protected access within their respective parent classes.

   Example:

   ```
   void foo(void) {
     // OK
     auto lam1 = [] __device__ { return 0; };
     {
       // OK
       auto lam2 = [] __device__ { return 0; };
       // OK
       auto lam3 = [] __device__ __host__ { return 0; };
     }
   }

   struct S1_t {
     S1_t(void) {
       // Error: cannot take address of enclosing function
       auto lam4 = [] __device__ { return 0; };
     }
   };

   class C0_t {
     void foo(void) {
       // Error: enclosing function has private access in parent class
       auto temp1 = [] __device__ { return 10; };
     }
     struct S2_t {
       void foo(void) {
         // Error: enclosing class S2_t has private access in its
         // parent class
         auto temp1 = [] __device__ { return 10; };
       }
     };
   };
   ```
5. It must be possible to take the address of the enclosing routine unambiguously, at the point where the extended lambda has been defined. This may not be feasible in some cases e.g. when a class typedef shadows a template type argument of the same name.

   Example:

   ```
   template <typename> struct A {
     typedef void Bar;
     void test();
   };

   template<> struct A<void> { };

   template <typename Bar>
   void A<Bar>::test() {
     /* In code sent to host compiler, nvcc will inject an
        address expression here, of the form:
        (void (A< Bar> ::*)(void))(&A::test))

        However, the class typedef 'Bar' (to void) shadows the
        template argument 'Bar', causing the address
        expression in A<int>::test to actually refer to:
        (void (A< void> ::*)(void))(&A::test))

        ..which doesn't take the address of the enclosing
        routine 'A<int>::test' correctly.
     */
     auto lam1 = [] __host__ __device__ { return 4; };
   }

   int main() {
     A<int> xxx;
     xxx.test();
   }
   ```
6. An extended lambda cannot be defined in a class that is local to a function.

   Example:

   ```
   void foo(void) {
     struct S1_t {
       void bar(void) {
         // Error: bar is member of a class that is local to a function.
         auto lam4 = [] __host__ __device__ { return 0; };
       }
     };
   }
   ```
7. The enclosing function for an extended lambda cannot have deduced return type.

   Example:

   ```
   auto foo(void) {
     // Error: the return type of foo is deduced.
     auto lam1 = [] __host__ __device__ { return 0; };
   }
   ```
8. \_\_host\_\_ \_\_device\_\_ extended lambdas cannot be generic lambdas.

   Example:

   ```
   void foo(void) {
     // Error: __host__ __device__ extended lambdas cannot be
     // generic lambdas.
     auto lam1 = [] __host__ __device__ (auto i) { return i; };

     // Error: __host__ __device__ extended lambdas cannot be
     // generic lambdas.
     auto lam2 = [] __host__ __device__ (auto ...i) {
                  return sizeof...(i);
                 };
   }
   ```
9. If the enclosing function is an instantiation of a function template or a member function template, and/or the function is a member of a class template, the template(s) must satisfy the following constraints:

   * The template must have at most one variadic parameter, and it must be listed last in the template parameter list.
   * The template parameters must be named.
   * The template instantiation argument types cannot involve types that are either local to a function (except for closure types for extended lambdas), or are private or protected class members.

   Example:

   ```
   template <typename T>
   __global__ void kern(T in) { in(); }

   template <typename... T>
   struct foo {};

   template < template <typename...> class T, typename... P1,
             typename... P2>
   void bar1(const T<P1...>, const T<P2...>) {
     // Error: enclosing function has multiple parameter packs
     auto lam1 =  [] __device__ { return 10; };
   }

   template < template <typename...> class T, typename... P1,
             typename T2>
   void bar2(const T<P1...>, T2) {
     // Error: for enclosing function, the
     // parameter pack is not last in the template parameter list.
     auto lam1 =  [] __device__ { return 10; };
   }

   template <typename T, T>
   void bar3(void) {
     // Error: for enclosing function, the second template
     // parameter is not named.
     auto lam1 =  [] __device__ { return 10; };
   }

   int main() {
     foo<char, int, float> f1;
     foo<char, int> f2;
     bar1(f1, f2);
     bar2(f1, 10);
     bar3<int, 10>();
   }
   ```

   Example:

   ```
   template <typename T>
   __global__ void kern(T in) { in(); }

   template <typename T>
   void bar4(void) {
     auto lam1 =  [] __device__ { return 10; };
     kern<<<1,1>>>(lam1);
   }

   struct C1_t { struct S1_t { }; friend int main(void); };
   int main() {
     struct S1_t { };
     // Error: enclosing function for device lambda in bar4
     // is instantiated with a type local to main.
     bar4<S1_t>();

     // Error: enclosing function for device lambda in bar4
     // is instantiated with a type that is a private member
     // of a class.
     bar4<C1_t::S1_t>();
   }
   ```
10. With Visual Studio host compilers, the enclosing function must have external linkage. The restriction is present because this host compiler does not support using the address of non-extern linkage functions as template arguments, which is needed by the CUDA compiler transformations to support extended lambdas.
11. With Visual Studio host compilers, an extended lambda shall not be defined within the body of an ‘if-constexpr’ block.
12. An extended lambda has the following restrictions on captured variables:

    * In the code sent to the host compiler, the variable may be passed by value to a sequence of helper functions before being used to direct-initialize the field of the class type used to represent the closure type for the extended lambda[25](#fn32).
    * A variable can only be captured by value.
    * A variable of array type cannot be captured if the number of array dimensions is greater than 7.
    * For a variable of array type, in the code sent to the host compiler, the closure type’s array field is first default-initialized, and then each element of the array field is copy-assigned from the corresponding element of the captured array variable. Therefore, the array element type must be default-constructible and copy-assignable in host code.
    * A function parameter that is an element of a variadic argument pack cannot be captured.
    * The type of the captured variable cannot involve types that are either local to a function (except for closure types of extended lambdas), or are private or protected class members.
    * For a \_\_host\_\_ \_\_device\_\_ extended lambda, the types used in the return or parameter types of the lambda expression’s `operator()` cannot involve types that are either local to a function (except for closure types of extended lambdas), or are private or protected class members.
    * Init-capture is not supported for \_\_host\_\_ \_\_device\_\_ extended lambdas. Init-capture is supported for \_\_device\_\_ extended lambdas, except when the init-capture is of array type or of type `std::initializer_list`.
    * The function call operator for an extended lambda is not constexpr. The closure type for an extended lambda is not a literal type. The constexpr and consteval specifier cannot be used in the declaration of an extended lambda.
    * A variable cannot be implicitly captured inside an if-constexpr block lexically nested inside an extended lambda, unless it has already been implicitly captured earlier outside the if-constexpr block or appears in the explicit capture list for the extended lambda (see example below).

    Example

    ```
    void foo(void) {
      // OK: an init-capture is allowed for an
      // extended __device__ lambda.
      auto lam1 = [x = 1] __device__ () { return x; };

      // Error: an init-capture is not allowed for
      // an extended __host__ __device__ lambda.
      auto lam2 = [x = 1] __host__ __device__ () { return x; };

      int a = 1;
      // Error: an extended __device__ lambda cannot capture
      // variables by reference.
      auto lam3 = [&a] __device__ () { return a; };

      // Error: by-reference capture is not allowed
      // for an extended __device__ lambda.
      auto lam4 = [&x = a] __device__ () { return x; };

      struct S1_t { };
      S1_t s1;
      // Error: a type local to a function cannot be used in the type
      // of a captured variable.
      auto lam6 = [s1] __device__ () { };

      // Error: an init-capture cannot be of type std::initializer_list.
      auto lam7 = [x = {11}] __device__ () { };

      std::initializer_list<int> b = {11,22,33};
      // Error: an init-capture cannot be of type std::initializer_list.
      auto lam8 = [x = b] __device__ () { };

      // Error scenario (lam9) and supported scenarios (lam10, lam11)
      // for capture within 'if-constexpr' block
      int yyy = 4;
      auto lam9 = [=] __device__ {
        int result = 0;
        if constexpr(false) {
          //Error: An extended __device__ lambda cannot first-capture
          //      'yyy' in constexpr-if context
          result += yyy;
        }
        return result;
      };

      auto lam10 = [yyy] __device__ {
        int result = 0;
        if constexpr(false) {
          //OK: 'yyy' already listed in explicit capture list for the extended lambda
          result += yyy;
        }
        return result;
      };

      auto lam11 = [=] __device__ {
        int result = yyy;
        if constexpr(false) {
          //OK: 'yyy' already implicit captured outside the 'if-constexpr' block
          result += yyy;
        }
        return result;
      };
    }
    ```
13. When parsing a function, the CUDA compiler assigns a counter value to each extended lambda within that function. This counter value is used in the substituted named type passed to the host compiler. Hence, whether or not an extended lambda is defined within a function should not depend on a particular value of `__CUDA_ARCH__`, or on `__CUDA_ARCH__` being undefined.

    Example

    ```
    template <typename T>
    __global__ void kernel(T in) { in(); }

    __host__ __device__ void foo(void) {
      // Error: the number and relative declaration
      // order of extended lambdas depends on
      // __CUDA_ARCH__
    #if defined(__CUDA_ARCH__)
      auto lam1 = [] __device__ { return 0; };
      auto lam1b = [] __host___ __device__ { return 10; };
    #endif
      auto lam2 = [] __device__ { return 4; };
      kernel<<<1,1>>>(lam2);
    }
    ```
14. As described above, the CUDA compiler replaces a `__device__` extended lambda defined in a host function with a placeholder type defined in namespace scope. Unless the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns true for the closure type of the extended lambda, the placeholder type does not define a `operator()` function equivalent to the original lambda declaration. An attempt to determine the return type or parameter types of the `operator()` function of such a lambda may therefore work incorrectly in host code, as the code processed by the host compiler will be semantically different than the input code processed by the CUDA compiler. However, it is OK to introspect the return type or parameter types of the `operator()` function within device code. Note that this restriction does not apply to `__host__ __device__` extended lambdas, or to `__device__` extended lambdas for which the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns true.

    Example

    ```
    #include <type_traits>
    const char& getRef(const char* p) { return *p; }

    void foo(void) {
      auto lam1 = [] __device__ { return "10"; };

      // Error: attempt to extract the return type
      // of a __device__ lambda in host code
      std::result_of<decltype(lam1)()>::type xx1 = "abc";


      auto lam2 = [] __host__ __device__  { return "10"; };

      // OK : lam2 represents a __host__ __device__ extended lambda
      std::result_of<decltype(lam2)()>::type xx2 = "abc";

      auto lam3 = []  __device__ () -> const char * { return "10"; };

      // OK : lam3 represents a __device__ extended lambda with preserved return type
      std::result_of<decltype(lam3)()>::type xx2 = "abc";
      static_assert( std::is_same_v< std::result_of<decltype(lam3)()>::type, const char *>);

      auto lam4 = [] __device__ (char x) -> decltype(getRef(&x)) { return 0; };
      // lam4's return type is not preserved because it references the operator()'s
      // parameter types in the trailing return type.
      static_assert( ! __nv_is_extended_device_lambda_with_preserved_return_type(decltype(lam4)), "" );
    }
    ```
15. For an extended device lambda:
    - Introspecting the parameter type of operator() is only supported in device code.
    - Introspecting the return type of operator() is supported only in device code, unless the trait function \_\_nv\_is\_extended\_device\_lambda\_with\_preserved\_return\_type() returns true.
16. If the functor object represented by an extended lambda is passed from host to device code (e.g., as the argument of a `__global__` function), then any expression in the body of the lambda expression that captures variables must be remain unchanged irrespective of whether the `__CUDA_ARCH__` macro is defined, and whether the macro has a particular value. This restriction arises because the lambda’s closure class layout depends on the order in which captured variables are encountered when the compiler processes the lambda expression; the program may execute incorrectly if the closure class layout differs in device and host compilation.

    Example

    ```
    __device__ int result;

    template <typename T>
    __global__ void kernel(T in) { result = in(); }

    void foo(void) {
      int x1 = 1;
      auto lam1 = [=] __host__ __device__ {
        // Error: "x1" is only captured when __CUDA_ARCH__ is defined.
    #ifdef __CUDA_ARCH__
        return x1 + 1;
    #else
        return 10;
    #endif
      };
      kernel<<<1,1>>>(lam1);
    }
    ```
17. As described previously, the CUDA compiler replaces an extended `__device__` lambda expression with an instance of a placeholder type in the code sent to the host compiler. This placeholder type does not define a pointer-to-function conversion operator in host code, however the conversion operator is provided in device code. Note that this restriction does not apply to `__host__ __device__` extended lambdas.

    Example

    ```
    template <typename T>
    __global__ void kern(T in) {
      int (*fp)(double) = in;

      // OK: conversion in device code is supported
      fp(0);
      auto lam1 = [](double) { return 1; };

      // OK: conversion in device code is supported
      fp = lam1;
      fp(0);
    }

    void foo(void) {
      auto lam_d = [] __device__ (double) { return 1; };
      auto lam_hd = [] __host__ __device__ (double) { return 1; };
      kern<<<1,1>>>(lam_d);
      kern<<<1,1>>>(lam_hd);

      // OK : conversion for __host__ __device__ lambda is supported
      // in host code
      int (*fp)(double) = lam_hd;

      // Error: conversion for __device__ lambda is not supported in
      // host code.
      int (*fp2)(double) = lam_d;
    }
    ```
18. As described previously, the CUDA compiler replaces an extended `__device__` or `__host__ __device__` lambda expression with an instance of a placeholder type in the code sent to the host compiler. This placeholder type may define C++ special member functions (e.g. constructor, destructor). As a result, some standard C++ type traits may return different results for the closure type of the extended lambda, in the CUDA frontend compiler versus the host compiler. The following type traits are affected: `std::is_trivially_copyable`, `std::is_trivially_constructible`, `std::is_trivially_copy_constructible`, `std::is_trivially_move_constructible`, `std::is_trivially_destructible`.

    Care must be taken that the results of these type traits are not used in `__global__` function template instantiation or in `__device__ / __constant__ / __managed__` variable template instantiation.

    Example

    ```
    template <bool b>
    void __global__ foo() { printf("hi"); }

    template <typename T>
    void dolaunch() {

    // ERROR: this kernel launch may fail, because CUDA frontend compiler
    // and host compiler may disagree on the result of
    // std::is_trivially_copyable() trait on the closure type of the
    // extended lambda
    foo<std::is_trivially_copyable<T>::value><<<1,1>>>();
    cudaDeviceSynchronize();
    }

    int main() {
    int x = 0;
    auto lam1 = [=] __host__ __device__ () { return x; };
    dolaunch<decltype(lam1)>();
    }
    ```

The CUDA compiler will generate compiler diagnostics for a subset of cases described in 1-12; no diagnostic will be generated for cases 13-17, but the host compiler may fail to compile the generated code.