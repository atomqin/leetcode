![](https://img-blog.csdnimg.cn/20200819094304960.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pcmFjbGVvbg==,size_16,color_FFFFFF,t_70#pic_center)

Trie，又称前缀树或字典树，用于判断字符串是否存在或者是否具有某种字符串前缀。
### 实现一个Trie
力扣[208](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)
```java
class Trie {
    class Node{
        boolean isEnd;
        Node[] next = new Node[26];
    }
    private Node root;
    /** Initialize your data structure here. */
    public Trie() {
        root = new Node();
    }
    
    /** Inserts a word into the trie. */
    public void insert(String word) {
        Node node = root;//root为空链接，不要更改
        for (char c : word.toCharArray()){
            if (node.next[c - 'a'] == null){
            //初始化一个新节点
                node.next[c - 'a'] = new Node();
            }
            node = node.next[c - 'a'];//迭代
        }
        node.isEnd = true;//单词末尾
    }
    
    /** Returns if the word is in the trie. */
    public boolean search(String word) {
        Node node = root;
        for (char c : word.toCharArray()){
            node = node.next[c - 'a'];
            if (node == null) return false;
        }
        return node.isEnd;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    public boolean startsWith(String prefix) {
        Node node = root;
        for (char c : prefix.toCharArray()){
            node = node.next[c - 'a'];
            if (node == null) return false;
        }
        return true;
    }
}
```
### 实现一个Trie，用来实现前缀和
[力扣677](https://leetcode-cn.com/problems/map-sum-pairs/)
```
Input: insert("apple", 3), Output: Null
Input: sum("ap"), Output: 3
Input: insert("app", 2), Output: Null
Input: sum("ap"), Output: 5
```
```java
class MapSum {
    class Node{
        int val;
        boolean isEnd;
        Node[] next;
        public Node(){
            this.val = val;
            this.isEnd = false;
            next = new Node[26];
        }
    }
    private Node root;
    /** Initialize your data structure here. */
    public MapSum() {
        root = new Node();
    }
    
    public void insert(String key, int val) {
        Node node = root;
        for (char c : key.toCharArray()){
            if (node.next[c - 'a'] == null){
                node.next[c - 'a'] = new Node();
            }
            node = node.next[c - 'a'];
        }
        node.isEnd = true;
        node.val = val;
    }
    
    public int sum(String prefix) {
        Node node = root;
        for (char c : prefix.toCharArray()){
            if (node.next[c - 'a'] == null) return 0;
            node = node.next[c - 'a'];
        }
        //此时到达前缀末尾
        return dfs(node);
    }
    private int dfs(Node node){
        if (node == null) return 0;
        int curSum = 0;
        if (node.isEnd == true) curSum += node.val;//插入单词的末尾
        //可以直接用next指代下一个连接的节点
        for (Node cur : node.next){
            curSum += dfs(cur);//递归
        }
        return curSum;
    }
}
```
