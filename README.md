## 字符串
### KMP
####  程序员小灰版本


==主字符串`str`和模式字符串`pattern`==
`next[]`数组记录的就是**最长可匹配子前缀**
如对于模式字符串 `GTGTGCF`，它的`next[]`数组为`0,0,0,1,2,3,0`
**注**:`next[0],next[1]`一定为 0，因为 0 或 1 个数字是没有**子**前缀的
`next[i]`中的`i`实际上可以看成是前面字符的长度
主字符串`str[i]`和模式字符串`pattern[j]`不等，则`j = next[j]`，相当于回溯到`next[j]`处
如`str = "GTGTGTGCF"`，当匹配到`i = 5`时，`str[5] = 'T',pattern[j] = C`，此时`str[i] != pattern[j]`，`j = next[5] = 3`;`pattern[3] = 'T' = str[5]`
<img src="https://img-blog.csdnimg.cn/20210425192528821.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pcmFjbGVvbg==,size_16,color_FFFFFF,t_70" alt="在这里插入图片描述" style="zoom:80%;" />   

```java
public class Solution {
      public static void main(String[] args) {
        Solution s = new Solution();
        String str = "GTGTGAGCTGGTGTGTGCFAA";
        String pattern = "GTGTGCF";
        System.out.println(s.KMP(str, pattern));
    }

    private int KMP(String str, String pattern){
        int j = 0;
        int[] next = NEXT(pattern);
        for (int i = 0; i < str.length(); i++) {
            //最长可匹配前缀，减少移动次数，好好体会
            while (j > 0 && str.charAt(i) != pattern.charAt(j)) {
                j = next[j];
            }
            if (str.charAt(i) == pattern.charAt(j)) {
                j++;
            }
            if (j == pattern.length()) {
                return i - j + 1;
            }
        }
        return -1;
    }

    private int[] NEXT(String pattern) {
        int[] next = new int[pattern.length()];
        int j = 0;
        for (int i = 2; i < pattern.length(); i++) {
            while (j != 0 && pattern.charAt(i - 1) != pattern.charAt(j)) {
                j = next[j];
            }
            if (pattern.charAt(i - 1) == pattern.charAt(j)) {
                j++;
            }
            next[i] = j;
        }
        return next;
    }
}
```
### 其他写法
next[i] 表示 i（包括i）之前最长相等的前后缀长度（其实就是j）

next[0]初始化为-1, j 从 -1 开始,
```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        def getNext(str):
            n = len(str)
            next = [0] * n
            j = -1
            next[0] = -1
            for i in range(1, n):
                while j >= 0 and str[i] != str[j + 1]:
                    j = next[j]
                if str[i] == str[j + 1]:
                    j += 1
                next[i] = j
            return next
        if len(needle) == 0:
            return 0
        next = getNext(needle)
        n = len(haystack)
        j = -1
        for i in range(0, n):
            while j >= 0 and haystack[i] != needle[j + 1]:
                j = next[j]
            if haystack[i] == needle[j + 1]:
                j += 1
            if j == len(needle) - 1:
                return i - j
        return -1
```
前缀表不减一构建next数组, next[0] = 0, j 从 0 开始
```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        def getNext(str):
            n = len(str)
            next = [0] * n
            j = 0
            
            for i in range(1, n):
                while j > 0 and str[i] != str[j]:
                    j = next[j - 1]
                if str[i] == str[j]:
                    j += 1
                next[i] = j
            return next
        if len(needle) == 0:
            return 0
        next = getNext(needle)
        n = len(haystack)
        j = 0
        for i in range(0, n):
            while j > 0 and haystack[i] != needle[j]:
                j = next[j - 1]
            if haystack[i] == needle[j]:
                j += 1
            if j == len(needle):
                return i - j + 1
        return -1
```
## LRU(Least Recently Used)
力扣 146 题”LRU缓存机制“

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

**哈希链表**`LinkedHashMap`:它是双向链表和哈希表的结合体

每次默认从链表头添加元素，链表尾消除元素，即左增右消（也可左消右增）
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
