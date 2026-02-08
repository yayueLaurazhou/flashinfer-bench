# device-graph-update

###### 6.2.8.7.7.1.3. Device Graph Update[](#device-graph-update "Permalink to this headline")

Device graphs can only be updated from the host, and must be re-uploaded to the device upon executable graph update in order for the changes to take effect. This can be achieved using the same methods outlined in the previous section. Unlike host graphs, launching a device graph from the device while an update is being applied will result in undefined behavior.