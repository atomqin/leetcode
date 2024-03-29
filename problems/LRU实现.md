```
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
 
进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？

示例：

输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
lRUCache.get(4);    // 返回 4
```

利用数据结构**哈希链表**`LinkedHashMap`:它是双向链表和哈希表的结合体

每次默认从链表头添加元素，链表尾消除元素，即左增右消（也可左消右增，默认应该是添加在表尾，如 python 中的 append 方法），这样的话越靠近表头的结点越“最近使用”，如`get`方法会将对应结点移到表头。当链表达到最大容量时，将表尾的结点移出链表

**注**：每次从链表中删除结点时，不要忘记将 map 中对应的键值对也去掉
```java
import java.util.HashMap;
class LRUCache {
    class Node{
        int key, val;
        Node prev, next;
        public Node(int key, int val){
            this.key = key;
            this.val = val;
        }
    }
    class DoubleList{
        //表头 表尾
        Node head, tail;
        int size;
        public DoubleList(){
            head = new Node(0, 0);
            tail = new Node(0, 0);
            head.next = tail;
            tail.prev = head;
            size = 0;
        }
        private void addFirst(Node x){
            x.prev = head;
            x.next = head.next;
            head.next.prev = x;
            head.next = x;
            size++;
        }
        private Node removeLast(){
            Node last = tail.prev;
            if (last == head) return null;
            remove(last);
            return last;
        }
        private void remove(Node x){
            x.prev.next = x.next;
            x.next.prev = x.prev;
            size--;
        }

        private int size(){
            return size;
        }
    }
    //最大容量
    private int cap;
    HashMap<Integer, Node> map;
    private DoubleList cache;

    public LRUCache(int capacity) {
        map = new HashMap<>();
        cache = new DoubleList();
        this.cap = capacity;
    }
    
    public int get(int key) {
        if (!map.containsKey(key)) {
            return -1;
        }
        Node x = map.get(key);
        map.remove(key);
        cache.remove(x);
        cache.addFirst(x);
        map.put(key, x);
        return x.val;
    }
    
    public void put(int key, int value) {
        
        Node x = new Node(key, value);
        if (map.containsKey(key)){
            // 旧键换新键
            cache.remove(map.get(key));
            map.remove(key);
            cache.addFirst(x);
            map.put(key, x);
        }else{
            //达到最大容量
            if (cap == cache.size()) {
                Node last = cache.removeLast();
                //这就是为什么结点要有key和val两个属性
                map.remove(last.key);
            }
            cache.addFirst(x);
            map.put(key, x);
        }
    }
}

/**
 * Your LRUCache object will be instantiated and called as such:
 * LRUCache obj = new LRUCache(capacity);
 * int param_1 = obj.get(key);
 * obj.put(key,value);
 */
```
- 也可以用自带的 `LinkedHashMap` 实现

**注**：默认的LinkedHashMap并不会移除旧元素，如果需要移除旧元素，则需要重写removeEldestEntry()方法设定移除策略；

```java
class LRUCache extends LinkedHashMap<Integer, Integer>{
    private int capacity;
    
    public LRUCache(int capacity) {
        // true:按访问顺序 false:按插入顺序
        super(capacity, 0.75F, true);
        this.capacity = capacity;
    }

    public int get(int key) {
        return super.getOrDefault(key, -1);
    }

    public void put(int key, int value) {
        super.put(key, value);
    }

    @Override
    protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
        return size() > capacity; 
    }
}
```
