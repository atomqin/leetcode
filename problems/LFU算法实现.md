- 核心思想：先考虑访问次数，在访问次数相同的情况下，再考虑缓存的时间。
- 写法一，利用 LinkedHashSet 存储相同freq下的 key 列表，表头即相同访问次数下需要删除的键值（默认加在表尾）
```java
class LFUCache {
/**
    1. 3个映射：keyToVal, keyToFreq, freqToKey -> HashMap<Integer, LinkedHashSet<Integer>>; 最小频率 minFreq
    2. get方法：key不存在返回-1；否则 freq 加 1，更新 keyToFreq; freqToKey 去掉freq列表里的key（去掉后列表如果为空，将freq也去掉）,添加到 freq+1 列表里
    3. put方法：
        3.1 key 若存在：更新key, freq + 1
        3.2 key 不存在：
            3.2.1 达到最大容量：去掉 minfreq 对应列表第一个key（最先被插入），更新三个映射
            3.2.2 else: freq = 1, minFreq更新为 1, 更新三个映射
    increaseFreq(key): freqToKey(freq).remove(key), 如果列表为空，去掉freq,如果freq恰好是minFreq, minFreq 加一，freqToKey(freq+1).add(key),更新keyToFreq
    removeMinFreqKey(): freqToKey(minFreq).remove(key),如果列表为空，去掉minFreq
 */
    HashMap<Integer, Integer> keyToVal;
    HashMap<Integer, Integer> keyToFreq;
    HashMap<Integer, LinkedHashSet<Integer>> freqToKey;
    int cap;
    int minFreq;

    public LFUCache(int capacity) {
        keyToVal = new HashMap<>();
        keyToFreq = new HashMap<>();
        freqToKey = new HashMap<>();
        this.cap = capacity;
        this.minFreq = 0;
    }
    
    public int get(int key) {
        if (!keyToVal.containsKey(key))
            return -1;
        increaseFreq(key);
        return keyToVal.get(key);
    }
    
    public void put(int key, int value) {
        if (this.cap <= 0) return;
        if (keyToVal.containsKey(key)){
            keyToVal.put(key, value);
            increaseFreq(key);
            return;
        }
        if (keyToVal.size() >= cap){
            removeMinFreqKey();
        }
        keyToVal.put(key, value);
        keyToFreq.put(key, 1);
        freqToKey.putIfAbsent(1, new LinkedHashSet<Integer>());
        freqToKey.get(1).add(key);
        this.minFreq = 1;
    }

    private void increaseFreq(int key){
        int freq = keyToFreq.get(key);
        keyToFreq.put(key, freq + 1);
        freqToKey.get(freq).remove(key);
        freqToKey.putIfAbsent(freq + 1, new LinkedHashSet<Integer>());
        freqToKey.get(freq + 1).add(key);
        if (freqToKey.get(freq).isEmpty()){
            freqToKey.remove(freq);
            if (freq == this.minFreq){
                this.minFreq++;
            }
        }
    }

    private void removeMinFreqKey(){
        int deletedKey = freqToKey.get(minFreq).iterator().next();
        keyToVal.remove(deletedKey);
        keyToFreq.remove(deletedKey);
        freqToKey.get(minFreq).remove(deletedKey);
        if (freqToKey.get(minFreq).isEmpty()){
            freqToKey.remove(minFreq);
        }
    }
}

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache obj = new LFUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```
- 自定义双向链表，存储键值对 Node，Node自带freq属性，双向链表表头添加结点，即相同freq下表尾结点需要删除

![Alt](https://pic.leetcode-cn.com/6295cf4a8078096ba9b049e17a6bf8b6be3079edbc8111363a3b3727cf37173e-2.jpg)

```java
class LFUCache {

    HashMap<Integer, Node> keysToNode;
    HashMap<Integer, DoubleList> freqToKeys;
    int capacity;
    int minFreq;

    public LFUCache(int capacity) {
        keysToNode = new HashMap<>();
        freqToKeys = new HashMap<>();
        this.capacity = capacity;
    }
    
    public int get(int key) {
        Node node = keysToNode.get(key);
        if (node == null){
            return -1;
        }
        IncreaseFreq(node);
        return node.val;
    }
    
    public void put(int key, int value) {
        if (capacity == 0) return;
        Node node = keysToNode.get(key);
        if (node != null){
            //注意这里要更新结点值，一开始忘了
            node.val = value;
            IncreaseFreq(node);
            keysToNode.put(key, node);
            return;
        }else {
            if (capacity == keysToNode.size()){
                removeMinFreq();
            }
            Node newNode = new Node(key, value);
            freqToKeys.putIfAbsent(1, new DoubleList());
            freqToKeys.get(1).add(newNode);
            keysToNode.put(key, newNode);
            minFreq = 1;
        }
    }

    private void IncreaseFreq(Node node){
        int freq = node.freq;
        DoubleList dl = freqToKeys.get(freq);
        dl.remove(node);
        if(dl.head.next == dl.tail){
            freqToKeys.remove(freq);
            if (freq == minFreq)
                minFreq++;
        }
        node.freq++;
        freqToKeys.putIfAbsent(freq + 1, new DoubleList());
        freqToKeys.get(freq + 1).add(node);
        // keysToNode.put(node.key, node);
    }

    private void removeMinFreq(){
        DoubleList dl = freqToKeys.get(minFreq);
        Node deletedNode = dl.tail.pre;
        dl.remove(deletedNode);
        if (dl == null)
            freqToKeys.remove(deletedNode.freq);
        keysToNode.remove(deletedNode.key);
    }

    class Node{
        int key, val;
        Node pre, next;
        int freq = 1;

        public Node(){}

        public Node(int key, int val){
            this.key = key;
            this.val = val;
        }
    }

    class DoubleList {
        Node head, tail;

        public DoubleList(){
            head = new Node();
            tail = new Node();
            head.next = tail;
            tail.pre = head;
        }
        // 表头添加元素
        private void add(Node node){
            node.next = head.next;
            node.pre = head;
            head.next.pre = node;
            head.next = node;
        }

        private void remove(Node node){
            node.pre.next = node.next;
            node.next.pre = node.pre;
        }
    }
}

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache obj = new LFUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```
- 存储频次的HashMap直接用双向链表代替
每一个频次用一个链表保存从新到旧的数据。

Cache中：firstLinkedList-->freq最大的链表-->...-->freq最小的链表-->lastLinkedList 每一个链表保存从新到旧的数据。

链表中：head-->最新的数据-->...-->最久的数据-->tail 当head.post指向tail的时候这个频次就没有数据此时删除链表。

Cache中：用Map记录Key-Node可以快速获取Node，Node中又直接存储了他的前节点、后节点以及所在频次链表， 所以可以对结点以及链表方便的进行操作。

```java
class LFUCache {
    Map<Integer, Node> cache;
    DoublyLinkedList firstLinkedList;
    DoublyLinkedList lastLinkedList;
    int size;
    int capacity;
    public LFUCache(int capacity) {
        cache = new HashMap<> (capacity);
        firstLinkedList = new DoublyLinkedList();
        lastLinkedList = new DoublyLinkedList();
        firstLinkedList.post = lastLinkedList;
        lastLinkedList.pre = firstLinkedList;
        this.capacity = capacity;
    }
    
    public int get(int key) {
        Node node = cache.get(key);
        if (node == null) {
            return -1;
        }
        freqInc(node);
        return node.value;
    }
    
    public void put(int key, int value) {
        if (capacity == 0) {
            return;
        }
        Node node = cache.get(key);
        if (node != null) {
            node.value = value;
            freqInc(node);
        } else {
            if (size == capacity) {
                // 如果缓存满了，删除lastLinkedList.pre链表中的tail.pre的Node，如果该链表中的元素删空了，则删掉该链表
                cache.remove(lastLinkedList.pre.tail.pre.key);
                lastLinkedList.removeNode(lastLinkedList.pre.tail.pre);
                size--;
                if (lastLinkedList.pre.head.post == lastLinkedList.pre.tail) {
                    removeDoublyLinkedList(lastLinkedList.pre);
                } 
            }
            Node newNode = new Node(key, value); 
            cache.put(key, newNode);
            if (lastLinkedList.pre.freq != 1) {
                DoublyLinkedList newDoublyLinedList = new DoublyLinkedList(1);
                addDoublyLinkedList(newDoublyLinedList, lastLinkedList.pre);
                newDoublyLinedList.addNode(newNode);
            } else {
                lastLinkedList.pre.addNode(newNode);
            }
            size++;
        }
    }

    void freqInc(Node node) {
        // 将node从原freq对应的链表里移除, 如果链表空了则删除链表,
        DoublyLinkedList linkedList = node.doublyLinkedList;
        DoublyLinkedList preLinkedList = linkedList.pre;
        linkedList.removeNode(node);
        if (linkedList.head.post == linkedList.tail) { 
            removeDoublyLinkedList(linkedList);
        }

        // 将node加入新freq对应的链表，若该链表不存在，则先创建该链表。
        node.freq++;
        if (preLinkedList.freq != node.freq) {
            DoublyLinkedList newDoublyLinedList = new DoublyLinkedList(node.freq);
            addDoublyLinkedList(newDoublyLinedList, preLinkedList);
            newDoublyLinedList.addNode(node);
        } else {
            preLinkedList.addNode(node);
        }
    }

    void addDoublyLinkedList(DoublyLinkedList newDoublyLinedList, DoublyLinkedList preLinkedList) {
        newDoublyLinedList.post = preLinkedList.post;
        newDoublyLinedList.post.pre = newDoublyLinedList;
        newDoublyLinedList.pre = preLinkedList;
        preLinkedList.post = newDoublyLinedList; 
    }

    void removeDoublyLinkedList(DoublyLinkedList doublyLinkedList) {
        doublyLinkedList.pre.post = doublyLinkedList.post;
        doublyLinkedList.post.pre = doublyLinkedList.pre;
    }
}
class Node {
    int key;
    int value;
    int freq = 1;
    Node pre;
    Node post;
    DoublyLinkedList doublyLinkedList;    

    public Node() {}
    
    public Node(int key, int value) {
        this.key = key;
        this.value = value;
    }
}

class DoublyLinkedList {
    int freq;
    DoublyLinkedList pre;
    DoublyLinkedList post;
    Node head;
    Node tail;

    public DoublyLinkedList() {
        head = new Node();
        tail = new Node();
        head.post = tail;
        tail.pre = head;
    }

    public DoublyLinkedList(int freq) {
        head = new Node();
        tail = new Node();
        head.post = tail;
        tail.pre = head;
        this.freq = freq;
    }

    void removeNode(Node node) {
        node.pre.post = node.post;
        node.post.pre = node.pre;
    }

    void addNode(Node node) {
        node.post = head.post;
        head.post.pre = node;
        head.post = node;
        node.pre = head;
        node.doublyLinkedList = this;
    }

}

/**
 * Your LFUCache object will be instantiated and called as such:
 * LFUCache obj = new LFUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```
