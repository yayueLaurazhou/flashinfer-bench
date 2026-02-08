# 18.5.25.2. Coroutine support

#### 18.5.25.2. Coroutine support[](#coroutine-support "Permalink to this headline")

Coroutines are not supported in device code. Uses of the `co_await`, `co_yield` and `co_return` keywords in the scope of a device function are diagnosed as error during device compilation.