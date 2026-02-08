# 6.2.8.8. Events

#### 6.2.8.8. Events[](#events "Permalink to this headline")

The runtime also provides a way to closely monitor the device’s progress, as well as perform accurate timing, by letting the application asynchronously record *events* at any point in the program, and query when these events are completed. An event has completed when all tasks - or optionally, all commands in a given stream - preceding the event have completed. Events in stream zero are completed after all preceding tasks and commands in all streams are completed.