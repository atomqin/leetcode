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

next[0]初始化为-1, j 从 -1 开始

![Alt](https://pic.leetcode-cn.com/1618845342-ydYJRp-9364346F937803F03CD1A0AE645EA0F1.jpg)

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
- 在实际编码时，通常会往原串和匹配串头部追加一个空格（哨兵）。**目的是让 j 下标从 0 开始，省去 j 从 -1 开始的麻烦**
```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        n, m = len(haystack), len(needle)
        if m == 0: return 0
        next = [0] * (m + 1)
        haystack = " " + haystack
        needle = " " + needle
        j = 1
        for i in range(2, m + 1):
            while j > 0 and needle[i] != needle[j + 1]:
                j = next[j]
            if needle[i] == needle[j + 1]:
                j += 1
            next[i] = j
        j = 0
        for i in range(1, n + 1):
            while j > 0 and haystack[i] != needle[j + 1]:
                j = next[j]
            if haystack[i] == needle[j + 1]:
                j += 1
            if j == m:
                return i - j
        return -y

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
