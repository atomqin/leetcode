### 2 两数相加

<font size=5>**题目描述**</font>

给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个**表示和**的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

**思路**：逆序存储，个十百千位从左到右，直接一个个相加即可

```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode curr = dummy;
        int sum = 0, carry = 0;
        // ListNode h1 = reverseListRecursion(l1);
        // ListNode h2 = reverseListRecursion(l2);
        while (l1 != null || l2 != null) {
            int x = (l1 == null) ? 0 : l1.val;
            int y = (l2 == null) ? 0 : l2.val;

            sum = x + y + carry;
            //进位
            carry = sum / 10;
            curr.next = new ListNode(sum % 10);
            curr = curr.next;
            if (l1 != null)
                l1 = l1.next;
            if (l2 != null)
                l2 = l2.next;
        }
        //最后有可能产生进位
        if (carry > 0)
            curr.next = new ListNode(carry);
        return dummy.next;
    }
}
```

**更进一步**：

正序存储呢，如

```
1->4->3
5->6->5
Output:7->0->8
```

思路：**反转链表**

**递归**

```java
 private ListNode reverseListRecursion(ListNode head){
        if(head == null || head.next == null) return head;
        ListNode newHead = reverseListRecursion(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
 }
```

**迭代**

```java
private static ListNode reverseListRecursion(ListNode head){
        ListNode curr = head, next = head;
        ListNode pre = null;
      
        while (next != null) {
            next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
}
```

### 4 寻找两个正序数组的中位数

**注**：要求复杂度 `log(m+n)`

**思路一**：转化为寻找第**K**个数，**K**为中间数

 假设我们要找第 `k` 小数，我们可以每次循环排除掉 `k/2` 个数。

[看这里]( [详细通俗的思路分析，多解法 - 寻找两个正序数组的中位数 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-w-2/) )

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        int midLeft = (m + n + 1) / 2;
        int mid1Right = (m + n + 2) / 2;
        //奇数偶数情况合并，奇数情况求两次相同的k
        return (findKth(nums1, 0, nums2, 0,  midLeft) + findKth(nums1, 0,nums2, 0,mid1Right)) * 0.5;
    }

    private int findKth(int[] nums1, int start1, int[] nums2, int start2, int k) {
        //数组剩余长度
        int len1 = nums1.length - start1;
        int len2 = nums2.length - start2;
        //让 len1 的长度小于 len2，这样就能保证如果有数组空了，一定是 len1 
        if (len1 > len2)
            return findKth(nums2, start2, nums1, start1, k);
        if (len1 == 0)
            return nums2[start2 + k - 1];
        if (k == 1)
            return Math.min(nums1[start1], nums2[start2]);
        // len1或len2可能小于 k/2, 此时对应i,j直接到数组末尾，因为是索引，所以要 -1
        int i = start1 + Math.min(len1, k / 2) - 1;
        int j = start2 + Math.min(len2, k / 2) - 1;
        if (nums1[i] < nums2[j])
            return findKth(nums1, i + 1, nums2, start2, k - (i - start1 + 1));
        else
            return findKth(nums1, start1, nums2, j + 1, k - (j - start2 + 1));
    }
}
```

  时间复杂度：每进行一次循环，我们就减少 k/2 个元素，所以时间复杂度是 O(log(k))，而 k=(m+n)/2，所以最终的复杂也就是 O(log(m+n。

空间复杂度：虽然我们用到了递归，但是可以看到这个递归属于**尾递归**，所以编译器不需要不停地堆栈

**思路二**：切割数组

- 两数组长度加起来是**偶数**情况下：当左右两部分数目相当时，中位数就是左边的最大值与右边最小值的平均值
- 两数组长度加起来是**奇数**情况下：当左部分数目比右部分数目多 1 时，中位数就是左边多出来的那个数

```java
class Solution {
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        //保证 m <= n， 因为 0 <= i <= m,要保证 0 <= j <= n的话，必须 m <= n
        //证明: j = (m + n + 1) / 2 - i <= (n + n + 1) / 2 - 0 = n
        // j >= (m + m + 1) / 2 - m = 0
        if (m > n)
            return findMedianSortedArrays(nums2, nums1);
        int iMin = 0, iMax = m;
        while (iMin <= iMax) {
            //二分查找
            int i = (iMin + iMax) / 2;
            //奇偶合并
            //奇数情况左半部分比右半部分多 1 个数字，i + j = m - i + n - j + 1, j = (m + n + 1) / 2 - i;
            //偶数情况 i + j = m - i + n - j, j = (m + n) / 2 - i
            // 对于偶数情况，(m + n + 1) / 2 == (m + n) / 2,所以可以统一写成 (m + n + 1) / 2
            // i 右移，j 就左移; i 左移， j 就右移
            int j = (m + n + 1) / 2 - i;
            // 左移
            if (i != 0 && j != n && nums1[i - 1] > nums2[j])
                iMax = i - 1;
            // 右移
            else if (i != m && j != 0 && nums2[j - 1] > nums1[i])
                iMin = i + 1;
            else {
                int maxLeft = 0;
                // j = (m + n + 1) / 2 - i, i 和 j 不可能同时为 0
                // 也不可能 i 为 m 的同时， j 为 n
                if (i == 0)
                    maxLeft = nums2[j - 1];
                else if (j == 0)
                    maxLeft = nums1[i - 1];
                else maxLeft = Math.max(nums1[i-1], nums2[j-1]);
                //奇数情况,中位数就是左半部分多出来的那个数
                if ((m + n) % 2 == 1)
                    return maxLeft;
                //偶数情况
                int minRight = 0;
                if (i == m)
                    minRight = nums2[j];
                else if (j == n)
                    minRight = nums1[i];
                else minRight = Math.min(nums1[i], nums2[j]);
                return (maxLeft + minRight) / 2.0;
            }
        }
        return 0.0;
    }

}
```



### 15 三数之和

![1617152890409](C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1617152890409.png)

```java
class Solution {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums.length < 3 || nums == null)
            return res;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 2; i++) {
            int lo = i + 1, hi = nums.length - 1;
            if (nums[i] > 0)
                return res;
            //最小值比0大，不用继续了
            int min = nums[i] + nums[lo] + nums[lo + 1];
            if (min > 0)
                break;
            //最大值比0小，本轮循环放弃
            int max = nums[i] + nums[hi] + nums[hi - 1];
            if (max < 0)
                continue;
            //第一个元素不能重复，不然相当于把nums[i-1]时的情况再走一遍
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            while (lo < hi) {
                if (nums[lo] + nums[hi] == -nums[i]) {
                    res.add(Arrays.asList(nums[i], nums[lo], nums[hi]));
                    while (lo < hi && nums[lo] == nums[lo + 1])//去重
                        lo++;
                    while (lo < hi && nums[hi] == nums[hi - 1])//去重
                        hi--;
                    lo++;
                    hi--;
                } else if (nums[lo] + nums[hi] < -nums[i])
                    lo++;
                else
                    hi--;
            }
        }
        return res;
    }
}
```



### 18 四数之和

![1617114893107](C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1617114893107.png)

```java
class Solution {
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if (nums == null || nums.length < 4)
            return res;
        int len = nums.length;
        Arrays.sort(nums);
        //定义4个指针i,j,lo,hi从0开始遍历，j从i+1开始遍历，留下lo和hi，lo指向j+1，hi指向数组最大值
        for (int i = 0; i < len - 3; i++) {
            //去重
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            //若最小值比target大，后面不用继续了
            int min1 = nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3];
            if (min1 > target)
                break;
            //当前循环的最大值比target小，本轮循环跳过
            int max1 = nums[i] + nums[len - 1] + nums[len - 2] + nums[len - 3];
            if (max1 < target)
                continue;
            // 第二层循环
            for (int j = i + 1; j < len - 2; j++) {
                if (j > i + 1 && nums[j] == nums[j - 1])
                    continue;
                int lo = j + 1, hi = len - 1;
                int min2 = nums[i] + nums[j] + nums[lo] + nums[lo + 1];
                if (min2 > target)
                    break;
                int max2 = nums[i] + nums[j] + nums[hi] + nums[hi - 1];
                if (max2 < target)
                    continue;
                while (lo < hi) {
                    int curr = nums[i] + nums[j] + nums[lo] + nums[hi];
                    if (curr == target) {
                        res.add(Arrays.asList(nums[i], nums[j], nums[lo], nums[hi]));
                        lo++;
                        while (lo < hi && nums[lo] == nums[lo - 1])
                            lo++;
                        hi--;
                        while (lo < hi && nums[hi] == nums[hi + 1])
                            hi--;
                    } else if (curr < target)
                        lo++;
                    else
                        hi--;
                }
            }
        }
        return res;
    }
}
```

### 42 接雨水

```
给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水

思路：对于第 i 根柱子，能接 min(left_max, right_max) - height[i] 雨水

```

```java
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        if (n == 0)
            return 0;
        int[] left_max = new int[n];
        int[] right_max = new int[n];
        int ans = 0;
        left_max[0] = height[0];
        right_max[n - 1] = height[n - 1];
        //正序
        for (int i = 1; i < n; i++) {
            // [0,i]最高柱子的高度
            left_max[i] = Math.max(left_max[i - 1], height[i]);
        }
        //逆序
        for (int j = n - 2; j >= 0; j--) {
            // [i, n-1]最高柱子的高度
            right_max[j] = Math.max(right_max[j + 1], height[j]);
        }
        for (int i = 0; i < n; i++) {
            ans += Math.min(left_max[i], right_max[i]) - height[i];
        }
        return ans;
    }
}

```

- 双指针法
- 我们只关心`min(l_max, r_max)`，不关心较大的那个是不是那边最大的值

```java
class Solution {
    public int trap(int[] height) {
        int n = height.length;
        if (n == 0)
            return 0;

        int ans = 0;
        int l_max = height[0], r_max = height[n - 1];
        int left = 0, right = n - 1;
        while (left <= right) {
            l_max = Math.max(l_max, height[left]);
            r_max = Math.max(r_max, height[right]);
            if (l_max < r_max) {
                ans += l_max - height[left];
                left++;
            } else {
                ans += r_max - height[right];
                right--;
            }
        }
        return ans;
    }
}

```

### 234 回文链表

- 递归（后序遍历）
- 时间复杂度 `O(N)`，空间复杂度为递归函数的堆栈`O(N)`

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        self.left = head
        def helper(node):
            if node is None:
                return True
            res = helper(node.next) and self.left.val == node.val
            self.left = self.left.next
            return res
        return helper(head)

```

- 双指针，原地反转链表
- 时间复杂度 `O(N)`，空间复杂度为`O(1)`

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        # 奇数长度需要在前进一位
        if fast:
            slow = slow.next
        # 翻转链表
        def reverse(root):
            pre, cur = None, root
            while cur:
                nextNode = cur.next
                cur.next = pre
                pre = cur
                cur = nextNode
            return pre
        left, right = head, reverse(slow)
        while right:
            if left.val != right.val:
                return False
            left = left.next
            right = right.next
        return True

```



## 二分查找

可结合`labuladong`公共号和`CS Notes`博客理解

**在寻找左右边界时，初始化 `right = nums.length`，因为如果初始化为长度减一，则当长度为1时，进入不了循环**

### 69 x的平方根

```java
实现 int sqrt(int x) 函数。

计算并返回 x 的平方根，其中 x 是非负整数。

由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

示例 1:

输入: 4
输出: 2
    
示例 2:

输入: 8
输出: 2
说明: 8 的平方根是 2.82842..., 
     由于返回类型是整数，小数部分将被舍去。

```

注意溢出，如果直接用 `mid * mid`判断，会溢出

**写法一** while( < )

注意最后 right 可能会大于结果，如 x = 8 时，循环结束后 right = 3，**需要减一**

```java
class Solution {
    public int mySqrt(int x) {
        if (x <= 1)
            return x;
        int left = 1, right = x;
        while (left < right) {
            int mid = (left + right) >>> 1;
            int sqrt = x / mid;
            if (mid < sqrt) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        //不要用 right * right > x 判断，会溢出
        if (right > x / right)
            return right - 1;
        return right;
    }
} 

```

**写法二** while( <= )

 对于 x = 8，它的开方是 2.82842...，最后应该返回 2 而不是 3。在循环条件为 left <= right 并且循环退出时，right 总是比 left 小 1，也就是说 right = 2，left = 3，因此最后的返回值应该为 right 而不是 left。 

```java
class Solution {
    public int mySqrt(int x) {
        if (x <= 1)
            return x;
        int left = 1, right = x;
        while (left <= right) {
            int mid = (left + right) >>> 1;
            int sqrt = x / mid;
            if (mid < sqrt) {
                left = mid + 1;
            } else if(mid > sqrt){
                right = mid - 1;
            } else {
                return mid;
            }
        }
        return right;
    }
}

```

### 540 有序数组中的单一元素

```java
给定一个只包含整数的有序数组，每个元素都会出现两次，唯有一个数只会出现一次，找出这个数。

示例 1:

输入: [1,1,2,3,3,4,4,8,8]
输出: 2
    
示例 2:

输入: [3,3,7,7,10,11,11]
输出: 10

```

如果索引为偶数时，`nums[index] == nums[index+1]`，说明左边没问题，将左边界移为`index+2`,若不等，有边界移为`index`;索引为奇数时，`nums[index] == nums[index-1]`，说明左边没问题，将左边界移为`index+1`，若不等，右边界移为`index`

```java
举例：[1 1 2 3 3]，index=2时，2和3不等，右边界变为索引2；index=1时，1和1相等，左边界变为2，退出循环

```

```java
class Solution {
    public int singleNonDuplicate(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int m = (l + r) >>> 1;
            if ((m & 1) == 0) {
                if (nums[m] == nums[m + 1]) {
                    l = m + 2;
                } else {
                    r = m;
                }
            } else if ((m & 1) == 1) {
                if (nums[m] == nums[m - 1]) {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
        }
        return nums[l];
    }
}

```

### 153 旋转数组的最小数字

```java
已知一个长度为 n 的数组，预先按照升序排列，经由 1 到 n 次 旋转 后，得到输入数组。例如，原数组 nums = [0,1,2,4,5,6,7] 在变化后可能得到：
若旋转 4 次，则可以得到 [4,5,6,7,0,1,2]
若旋转 7 次，则可以得到 [0,1,2,4,5,6,7]
注意，数组 [a[0], a[1], a[2], ..., a[n-1]] 旋转一次 的结果为数组 [a[n-1], a[0], a[1], a[2], ..., a[n-2]] 。

给你一个元素值 互不相同 的数组 nums ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。
 
示例 1：

输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组。

```

写法一：将中间数字与右端比较

```java
class Solution {
    public int findMin(int[] nums) {
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int m = (l + r) / 2;
            if (nums[m] > nums[r]) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        return nums[l];
    }
}

```

写法二：将中间数字与左端比较，**注意**：只有当`nums[m] > nums[m+1]`时，左边界`l`才能变为`m + 1`

```java
例如：[4 5 6 7 0 1 2] ，第一次循环，7 > 4,l = ‘0的索引4’错误，要比较 7 和 0 的大小，才能变换左边界

```

```java
class Solution {
    public int findMin(int[] nums) {
        if (nums[nums.length - 1] > nums[0])
            return nums[0];
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int m = (l + r) / 2;
            if (nums[m] < nums[l]) {
                r = m;
            } else {
                if (nums[m] > nums[m + 1]) {
                    return nums[m + 1];
                } else {
                    l = m + 1;
                }
            }
        }
        return nums[l];
    }
}

```

### 34 在排序数组中查找元素的第一个位置和最后一个位置

**思路**：写两个二分查找函数，分别查找元素的左边界和右边界

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] ans = new int[] { -1, -1 };
        if (nums.length == 0)
            return ans;
        int first = findFirst(nums, target);
        int last = findLast(nums, target);
        if (first == -1 || last == -1)
            return ans;
        return new int[]{first,last};
    }
    //左边界
    private int findFirst(int[] nums, int target) {
        int l = 0, r = nums.length;//可以减一，后面if(nums[r] != target) return -1
        while (l < r) {
            int m = (l + r) >>> 1;
            if (nums[m] < target) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        if (l == nums.length || nums[r] != target)
            return -1;
        return r;
    }
    //右边界
    private int findLast(int[] nums, int target) {
        int l = 0, r = nums.length;//不能减一，不然最后一个元素无法判断
        while (l < r) {
            int m = (l + r) >>> 1;
            if (nums[m] > target) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        if (l == 0 || nums[l - 1] != target)
            return -1;
        return l - 1;
    }
}

```

**思路二**： 我们将寻找 target 最后一个位置，转换成寻找 target+1 第一个位置，再往前移动一个位置。这样我们只需要实现一个二分查找代码即可。 

**搜索`val`的左侧边界时，若`val`不存在，得到的恰好是比`val`大的最小元素索引**

```java
class Solution {
    public int[] searchRange(int[] nums, int target) {
        int[] ans = new int[]{ -1, -1 };
        if (nums.length == 0)
            return ans;
        int first = findFirst(nums, target);
        int last = findFirst(nums, target + 1);
        //把判断放到这，不然last没法用
        if (first == nums.length || nums[first] != target)
            return ans;
        return new int[]{first,last - 1};
    }
    //左边界
    private int findFirst(int[] nums, int target) {
        int l = 0, r = nums.length;//注意初始化的值，防备长度为1的情况
        while (l < r) {
            int m = (l + r) >>> 1;
            if (nums[m] < target) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        //不急着判断 -1
        return r;
    }
}

```

## 字符串匹配

### KMP算法

####  程序员小灰版本

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1619350586866.png" alt="1619350586866" style="zoom:80%;" />
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

## 滑动窗口问题

### 3 [无重复字符的最长子串]

遇到重复字符，左光标直接跳到这个字符上次出现的位置后面

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0)
            return 0;
        Map<Character, Integer> window = new HashMap<>();
        int left = 0, right = 0;
        int len = 0;
        while (right < s.length()) {
            char x = s.charAt(right);
            if (window.containsKey(x)) {
                //注意这么一种情况，如：tmmzuxt，右光标到最后一个t时，左光标应该在索引2处，而不是第一个t后面
                left = Math.max(left, window.get(x) + 1);
            }
            window.put(x, right);
            right++;
            len = Math.max(len, right - left);
        }
        return len;
    }
}

```

**用数组也行**

```java
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0 || s.length() == 1)
            return s.length();
        int[] window = new int[128];
        // Arrays.fill(window, -1);
        int left = 0, right = 0;
        int len = 0;
        while (right < s.length()) {
            char x = s.charAt(right);
            if(window[x] > 0)
                left = Math.max(left, window[x]);
            //右侧窗口下一个位置，位置从1开始，避免第一个字符索引为0，不好用window[x]>0判断是否重复
            window[x] = right + 1;
            right++;
            len = Math.max(len, right - left);
        }
        return len;
    }
}

```

### 30 串联所有单词的子串

![1617089065641](C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1617089065641.png)

以 示例一 为例，单个单词长度为3，窗口从索引 0,1,2 处每次右移三位

```java
class Solution {
    public List<Integer> findSubstring(String s, String[] words) {
        List<Integer> res = new ArrayList<>();
        Map<String, Integer> wordsMap = new HashMap<>();
        if (s.length() == 0 || words.length == 0) return res;
        for (String word: words) {
            // 主串s中没有这个单词，直接返回空
            // if (s.indexOf(word) < 0) return res;
            // map中保存每个单词，和它出现的次数
            wordsMap.put(word, wordsMap.getOrDefault(word, 0) + 1);
        }
        // 每个单词的长度， 总长度
        int oneLen = words[0].length(), wordsLen = oneLen * words.length;
        // 主串s长度小于单词总和，返回空
        if (wordsLen > s.length()) return res;
        // 只讨论从0，1，...， oneLen-1 开始的子串情况，
        // 每次进行匹配的窗口大小为 wordsLen，每次后移一个单词长度，由左右窗口维持当前窗口位置
        for (int i = 0; i < oneLen; ++i) {
            // 左右窗口
            int left = i, right = i, count = 0;
            // 统计每个符合要求的word
            Map<String, Integer> subMap = new HashMap<>();
            // 右窗口不能超出主串长度
            while (right + oneLen <= s.length()) {
                // 得到一个单词
                String word = s.substring(right, right + oneLen);
                // 右窗口右移
                right += oneLen;
                // words[]中没有这个单词，那么当前窗口肯定匹配失败，直接右移到这个单词后面
                if (!wordsMap.containsKey(word)) {
                    left = right;
                    // 窗口内单词统计map清空，重新统计
                    subMap.clear();
                    // 符合要求的单词数清0
                    count = 0;
                } else {
                    // 统计当前子串中这个单词出现的次数
                    subMap.put(word, subMap.getOrDefault(word, 0) + 1);
                    count++;
                    // 如果这个单词出现的次数大于words[]中它对应的次数，又由于每次匹配和words长度相等的子串
                    // 如 ["foo","bar","foo","the"]  "| foobarfoobar| foothe"
                    // 第二个bar虽然是words[]中的单词，但是次数超了，那么右移一个单词长度后 "|barfoobarfoo|the"
                    // bar还是不符合，所以直接从这个不符合的bar之后开始匹配，也就是将这个不符合的bar和它之前的单词(串)全移出去
                    while (subMap.getOrDefault(word, 0) > wordsMap.getOrDefault(word, 0)) {
                        // 从当前窗口字符统计map中删除从左窗口开始到数量超限的所有单词(次数减一)
                        String w = s.substring(left, left + oneLen);
                        subMap.put(w, subMap.getOrDefault(w, 0) - 1);
                        // 符合的单词数减一
                        count--;
                        // 左窗口位置右移
                        left += oneLen;
                    }
                    // 当前窗口字符串满足要求
                    if (count == words.length) res.add(left);
                }
            }
        }
        return res;
    }
}

```



### 76 最小覆盖子串

给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 "" 。

注意：如果 s 中存在这样的子串，我们保证它是唯一的答案。

``` 示例 1：
示例 1：
输入：s = "ADOBECODEBANC", t = "ABC"
输出："BANC"

```

```java
class Solution {
    public String minWindow(String s, String t) {
        // 字母最大ASCII码小于128
        int[] need = new int[128];
        int[] window = new int[128];
        int left = 0, right = 0;
        int valid = 0;// 窗口中符合条件字符数
        int len = Integer.MAX_VALUE;
        String str = "";
        for (char c : t.toCharArray()) {
            need[c]++;
        }
        int n = 0;// t中不同字符数
        for (int i : need) {
            if (i == 0)
                continue;
            n++;
        }
        while (right < s.length()) {
            char x = s.charAt(right++);// 右侧窗口右移
            if (need[x] > 0) {
                window[x]++;
                if (window[x] == need[x])
                    valid++;
            }
            while (valid == n) {
                // 更新最小覆盖子串长度
                if (right - left < len) {
                    len = right - left;
                    str = s.substring(left, right);
                }
                char y = s.charAt(left++);// 左侧窗口右移
                if (need[y] > 0) {
                    window[y]--;
                    if (window[y] < need[y]) {
                        valid--;
                    }
                    // 和上面窗右侧右移对称
                    // if (window[y] == need[y]) {
                    //     valid--;
                    // }
                    // window[y]--;
                }
            }
        }
        return str;
    }
}

```



### 567 字符串排列

给定两个字符串 `s1` 和 `s2`，写一个函数来判断 `s2` 是否包含 `s1` 的排列。

换句话说，第一个字符串的排列之一是第二个字符串的 **子串** 。

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        if (s1.isEmpty())
            return false;
        int[] window = new int[128];
        int[] s1_map = new int[128];
        int left = 0, right = 0;
        int valid = 0;
        int n = 0;
        for (char c : s1.toCharArray()) {
            s1_map[c]++;
        }
        for (int i : s1_map) {
            if (i == 0)
                continue;
            n++;
        }
        while (right < s2.length()) {
            char x = s2.charAt(right++);
            if (s1_map[x] > 0) {
                window[x]++;
                if (window[x] == s1_map[x]) {
                    valid++;
                }
            }
            while (right - left >= s1.length()) {
                if (valid == n)
                    return true;
                char y = s2.charAt(left++);
                if (s1_map[y] > 0) {
                    if (window[y] == s1_map[y]) {
                        valid--;
                    }
                    window[y]--;
                }
            }
        }
        return false;
    }
}

```

用数组

```java
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length())
            return false;
        int[] window = new int[128];
        int[] s1_map = new int[128];
        
        //窗口长度为 s1.length()
        for (int i = 0; i < s1.length(); i++) {
            s1_map[s1.charAt(i)]++;
            window[s2.charAt(i)]++;
        }
        if (Arrays.equals(window, s1_map))
            return true;
        int left = 0, right = s1.length();
        while (right < s2.length()) {
            window[s2.charAt(right)]++;
            right++;
            window[s2.charAt(left)]--;
            left++;
            if (Arrays.equals(window, s1_map))
                return true;
        }
        return false;
    }
}

```

### 438 [找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/description/)

![1617021366707](C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1617021366707.png)

**直接套用模板**

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        //都是小写字母
        int[] window = new int[26];
        int[] p_map = new int[26];
        int left = 0, right = 0;
        int valid = 0;
        int n = 0;
        List<Integer> res = new ArrayList<>();
        for (char c : p.toCharArray()) {
            p_map[c - 'a']++;
        }
        for (int i : p_map) {
            if (i == 0)
                continue;
            n++;
        }
        while (right < s.length()) {
            char x = s.charAt(right);
            right++;
            if (p_map[x - 'a'] > 0) {
                window[x - 'a']++;
                if (window[x - 'a'] == p_map[x - 'a']) {
                    valid++;
                }
            }
            if (right - left == p.length()) {
                if (valid == n)
                    res.add(left);
                char y = s.charAt(left);
                left++;
                if (p_map[y - 'a'] > 0) {
                    if (window[y - 'a'] == p_map[y - 'a']) {
                        valid--;
                    }
                    window[y - 'a']--;
                }
            }
        }
        return res;
    }
}

```

或者

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        if (s.length() < p.length())
            return new ArrayList<>();
        List<Integer> res = new ArrayList<>();
        int[] window = new int[26];
        int[] p_map = new int[26];
        int left = 0, right = p.length();
        
        for (int i = 0; i < p.length(); i++) {
            p_map[p.charAt(i) - 'a']++;
            window[s.charAt(i) - 'a']++;
        }
        if (Arrays.equals(p_map, window))
            res.add(left);
        while (right < s.length()) {
            window[s.charAt(right) - 'a']++;
            right++;
            window[s.charAt(left) - 'a']--;
            left++;
            if (Arrays.equals(p_map, window))
                res.add(left);
        }
        return res;
    }
}

```

或者当`window`中的某一字符数超过`p_map`中对应的字符数时，缩减窗口

```java
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        if (s.length() < p.length())
            return new ArrayList<>();
        List<Integer> res = new ArrayList<>();
        int[] window = new int[26];
        int[] p_map = new int[26];
        int left = 0, right = 0;
        
        for (int i = 0; i < p.length(); i++) {
            p_map[p.charAt(i) - 'a']++;
        }
        while (right < s.length()) {
            window[s.charAt(right) - 'a']++;
            //如要求“abc”,给定“bacebcda”,到第二个b时，收缩窗口，直到索引4的“b"为止
            while (window[s.charAt(right) - 'a'] > p_map[s.charAt(right) - 'a']) {
                window[s.charAt(left) - 'a']--;
                left++;
            }
            //放下面，不然影响上面while语句的判断条件
            right++;
            //说明连续字符串符合条件，如要求“abc”,给出字符串为"bacebcda",前三个满足条件，后面的”bcda“就不满足条件
            if (right - left == p.length())
                res.add(left);
        }
        return res;
    }
}

```

## 分治

### 95 不同的二叉搜索树II

![1617671710928](C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1617671710928.png)

#### 分治

```java
/* public class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;

      TreeNode() {
      }

      TreeNode(int val) {
         this.val = val;
      }

      TreeNode(int val, TreeNode left, TreeNode right) {
         this.val = val;
         this.left = left;
         this.right = right;
      }
   } */
class Solution {
    public List<TreeNode> generateTrees(int n) {
        if (n < 1)
            return new ArrayList<>();
        return helper(1, n);
    }

    private List<TreeNode> helper(int start, int end) {
        //不能定义成全局变量
        List<TreeNode> list = new ArrayList<>();
        if (start > end) {
            list.add(null);
            return list;
        }
        /*
         * if (start == end) { list.add(new TreeNode(start)); return list; }
         */
        for (int i = start; i <= end; i++) {
            // 想想为什么这行不能放在这里，而放在下面？
            // TreeNode root = new TreeNode(i);
            List<TreeNode> leftSubTree = helper(start, i - 1);
            List<TreeNode> rightSubTree = helper(i + 1, end);
            for (TreeNode left : leftSubTree) {
                for (TreeNode right : rightSubTree) {
                    TreeNode root = new TreeNode(i);
                    root.left = left;
                    root.right = right;
                    list.add(root);
                }
            }
        }
        return list;
    }

}

```

[这里](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/solution/cong-gou-jian-dan-ke-shu-dao-gou-jian-suo-you-shu-/)关于`TreeNode root = new TreeNode(i)`的放置的位置问题
如果这行代码放置在注释的地方，会造成一个问题，就是以当前为 root 根结点的树个数
`num = left.size() * right.size() > 1`时，`num`棵子树会共用这个 root 结点，在下面两层 for 循环中，root的左右子树一直在更新，如果每次不新建一个root，就会导致`num`个 root 为根节点的树都相同。

关于如果当前子树为空，不加null行不行的问题
显然，如果一颗树的左子树为空，右子树不为空，要正确构建所有树，依赖于对左右子树列表的遍历，也就是上述代码两层for循环的地方，如果其中一个列表为空，那么循环都将无法进行。

#### 动态规划I

[看这里](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-2-7/)

```java
class Solution {
    public List<TreeNode> generateTrees(int n) {
        ArrayList<TreeNode>[] dp = new ArrayList[n + 1];
        dp[0] = new ArrayList<TreeNode>();// 如果left或right为0，就会出现空指针异常
        if (n == 0)
           return dp[0];
        dp[0].add(null);// 如果不加null，那么后面当left或right为0时，就不会执行foreach循环。而一旦left不执行，right也会被跳过
        for (int len = 1; len <= n; len++) {
           dp[len] = new ArrayList<TreeNode>();
           for (int root = 1; root <= len; root++) {
              int left = root - 1;
              int right = len - root;
              for (TreeNode leftTree : dp[left]) {
                 for (TreeNode rightTree : dp[right]) {
                    TreeNode treeRoot = new TreeNode(root);
                    treeRoot.left = leftTree;
                    treeRoot.right = clone(rightTree, root);
                    dp[len].add(treeRoot);
                 }
              }
           }
        }
        return dp[n];
     }
     /*
        假设n为5，root是3，那么左边有2个节点，所以去dp[2]里面找，右边也有两个节点4,5。所以还去dp[2]里面找。
        因为只有dp[2]里面都是2个节点的数。但是dp[2]中的数只有1和2，我们要的是4,5。
        我们不妨将1,2加上root，你会发现正好是4,5。
        所以得到结论，左子树的值直接找前面节点数一样的dp索引，右子树的值也找前面一样的dp索引,
        但是你需要加上root才能保证val是你需要的，所以右子树要重新创建，不然会破坏前面的树。
    */
     // 如果dp[left]里有两种可能，dp[right]里有三种可能，那么总共有6种可能。
  	//实现树的复制，加上偏差
     private TreeNode clone(TreeNode n, int offset) {
        if (n == null)
           return null;
        TreeNode root = new TreeNode(n.val + offset);
        root.left = clone(n.left, offset);
        root.right = clone(n.right, offset);
        return root;
     }
}

```

#### 动态规划II

```java
考虑 [] 的所有解
null

考虑 [ 1 ] 的所有解
1

考虑 [ 1 2 ] 的所有解
  2
 /
1

 1
  \
   2

考虑 [ 1 2 3 ] 的所有解
    3
   /
  2
 /
1

   2
  / \
 1   3
    
     3
    /
   1
    \
     2
       
   1
     \
      3
     /
    2
    
  1
    \
     2
      \
       3


```

仔细分析，可以发现一个规律。首先我们每次新增加的数字大于之前的所有数字，所以新增加的数字出现的位置只可能是根节点或者是根节点的右孩子，右孩子的右孩子，右孩子的右孩子的右孩子等等，总之一定是右边。其次，新数字所在位置原来的子树，改为当前插入数字的左孩子即可，因为插入数字是最大的。

```java
对于下边的解 
  2
 /
1

然后增加 3
1.把 3 放到根节点
    3
   /
  2
 /
1

2. 把 3 放到根节点的右孩子
   2
  / \
 1   3
 
对于下边的解
 1
  \
   2

然后增加 3
1.把 3 放到根节点
     3
    /
   1
    \
     2
       
2. 把 3 放到根节点的右孩子，原来的子树作为 3 的左孩子       
      1
        \
         3
        /
      2

3. 把 3 放到根节点的右孩子的右孩子
  1
    \
     2
      \
       3

```



```java
class Solution {
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode> pre = new ArrayList<TreeNode>();
        if (n == 0) {
            return pre;
        }
        //不加这句无法进入循环，n = 1 时，insert.left = null;
        pre.add(null);
        //每次增加一个数字
        for (int i = 1; i <= n; i++) {
            List<TreeNode> cur = new ArrayList<TreeNode>();
            //遍历之前的所有解
            for (TreeNode root : pre) {
                //插入到根节点
                TreeNode insert = new TreeNode(i);
                insert.left = root;
                cur.add(insert);
                //插入到右孩子，右孩子的右孩子...最多找 i - 1 次孩子,如 i = 4 时,最多往右遍历3次就到空结点了,结点4可以放到遍历3次过后的空结点上
                for (int j = 0; j < i - 1; j++) {
                    TreeNode root_copy = treeCopy(root); //复制当前的树
                    TreeNode right = root_copy; //找到要插入右孩子的位置
                    // int k = 0;
                    //先沿右结点遍历 j 次到插入节点的地方，从当前根节点的右子节点开始，遍历1次，直到遍历i-1次到达空节点
                    for (int k = 0; k < j; k++) {
                        if (right == null)
                            break;
                        right = right.right;
                    }
                    //到达 null 提前结束
                    if (right == null)
                        break;
                    //保存当前右孩子的位置的子树作为插入节点的左孩子
                    TreeNode rightTree = right.right;
                    insert = new TreeNode(i);
                    right.right = insert; //右孩子是插入的节点
                    insert.left = rightTree; //插入节点的左孩子更新为插入位置之前的子树
                    //加入结果中
                    cur.add(root_copy);
                }
            }
            pre = cur;

        }
        return pre;
    }

    private TreeNode treeCopy(TreeNode root) {
        if (root == null) {
            return root;
        }
        TreeNode newRoot = new TreeNode(root.val);
        newRoot.left = treeCopy(root.left);
        newRoot.right = treeCopy(root.right);
        return newRoot;
    }
}

```

### 241 给表达式加括号

**示例**

```java
输入: "2-1-1"
输出: [0, 2]
解释: 
((2-1)-1) = 0 
(2-(1-1)) = 2

```

#### 分治

```java
class Solution {
    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer> res = new ArrayList<>();
        //判断是否是数字
        /* if (!expression.contains("+") && !expression.contains("-") && !expression.contains("*")) {
            res.add(Integer.valueOf(expression));
            return res;
        } */
        for (int i = 0; i < expression.length(); i++) {
            char c = expression.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                for (int left : diffWaysToCompute(expression.substring(0, i))) {
                    for (int right : diffWaysToCompute(expression.substring(i + 1))) {
                        if (c == '+') {
                            res.add(left + right);
                        } else if (c == '-') {
                            res.add(left - right);
                        } else if (c == '*') {
                            res.add(left * right);
                        }
                    }
                }
            }
        }
        //是数字的话
        if (res.size() == 0)
            res.add(Integer.valueOf(expression));
        return res;
    }
}

```

由于递归是两个分支，所以会有一些的解进行了重复计算，我们可以通过 `memoization `技术，前边很多题都用过了，一种空间换时间的方法。

将递归过程中的解保存起来，如果第二次递归过来，直接返回结果即可，无需重复递归。

将解通过 map 存储，其中，key 存储函数入口参数的字符串，value 存储当前全部解的一个 List 。

```java
class Solution {
    //添加map
    Map<String, List<Integer>> map = new HashMap<>();
    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer> res = new ArrayList<>();
        /* if (!expression.contains("+") && !expression.contains("-") && !expression.contains("*")) {
            res.add(Integer.valueOf(expression));
            return res;
        } */
        //如果已经有当前解了，直接返回
        if (map.containsKey(expression))
            return map.get(expression);
        for (int i = 0; i < expression.length(); i++) {
            char c = expression.charAt(i);
            if (c == '+' || c == '-' || c == '*') {
                for (int left : diffWaysToCompute(expression.substring(0, i))) {
                    for (int right : diffWaysToCompute(expression.substring(i + 1))) {
                        if (c == '+') {
                            res.add(left + right);
                        } else if (c == '-') {
                            res.add(left - right);
                        } else if (c == '*') {
                            res.add(left * right);
                        }
                    }
                }
            }
        }
        //存到map
        map.put(expression, res);
        //是数字的话
        if (res.size() == 0) {
            res.add(Integer.valueOf(expression));
            map.put(expression, res);
        }
        return res;
    }
}

```

#### 动态规划

[看这里](https://leetcode-cn.com/problems/different-ways-to-add-parentheses/solution/xiang-xi-tong-su-de-si-lu-fen-xi-duo-jie-fa-by-5-5/)

`dp[i][j]`表示从索引 `i`到`j`的所有解

```java
class Solution {
    public List<Integer> diffWaysToCompute(String input) {
        List<Integer> numList = new ArrayList<>();
        List<Character> opList = new ArrayList<>();
        char[] array = input.toCharArray();
        int num = 0;
        for (int i = 0; i < array.length; i++) {
            if (isOperation(array[i])) {
                numList.add(num);
                num = 0;
                opList.add(array[i]);
            }else
            	num = num * 10 + array[i] - '0';
        }
        //最后还有一个数字没加入
        numList.add(num);
        int N = numList.size(); // 数字的个数
    
        // 一个数字
        ArrayList<Integer>[][] dp = new ArrayList[N][N];
        for (int i = 0; i < N; i++) {
            ArrayList<Integer> result = new ArrayList<>();
            result.add(numList.get(i));
            dp[i][i] = result;
        }
        // 2 个数字到 N 个数字
        for (int n = 2; n <= N; n++) {
            // 开始下标
            for (int i = 0; i < N - 1; i++) {
                // 结束下标
                int j = i + n - 1;
                if (j >= N) {
                    break;
                }
                ArrayList<Integer> result = new ArrayList<>();
                // 分成 i ~ s 和 s+1 ~ j 两部分
                for (int s = i; s < j; s++) {
                    ArrayList<Integer> result1 = dp[i][s];
                    ArrayList<Integer> result2 = dp[s + 1][j];
                    for (int x = 0; x < result1.size(); x++) {
                        for (int y = 0; y < result2.size(); y++) {
                            // 第 s 个数字下标对应是第 s 个运算符
                            char op = opList.get(s);
                            result.add(calculate(result1.get(x), op, result2.get(y)));
                        }
                    }
                }
                dp[i][j] = result;
    
            }
        }
        return dp[0][N-1];
    }
    
    private int calculate(int num1, char c, int num2) {
        switch (c) {
            case '+':
                return num1 + num2;
            case '-':
                return num1 - num2;
            case '*':
                return num1 * num2;
        }
        return -1;
    }
    
    private boolean isOperation(char c) {
        return c == '+' || c == '-' || c == '*';
    }
}

```

### 53 最大子序和

**题目描述**

```
输入：nums = [-2,1,-3,4,-1,2,1,-5,4]
输出：6
解释：连续子数组 [4,-1,2,1] 的和最大，为 6 。

```

#### 动态规划

`dp[i]`表示以`nums[i]`结尾的最大子序和，`dp[i-1] `若大于等于0，则有增益，`dp[i] = dp[i - 1] + nums[i]`，不然 `dp[i] = nums[i]`

```java
class Solution {
    public int maxSubArray(int[] nums) {
        if (nums.length == 1)
            return nums[0];
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int ans = nums[0];
        for (int i = 1; i < nums.length; i++) {
            if (dp[i - 1] >= 0)
                dp[i] = dp[i - 1] + nums[i];
            else
                dp[i] = nums[i];
            ans = Math.max(ans, dp[i]);
        }
        //另一种写法
        /*for (int i = 1; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            ans = Math.max(ans, dp[i]);
        }*/
        return ans;
    }
}

```

`dp[i]`只与前一个状态有关，优化数组为一个变量`ans`

```java
class Solution {
    public int maxSubArray(int[] nums) {
        if (nums.length == 1)
            return nums[0];
        int ans = Integer.MIN_VALUE;
        int sum = 0;
        for (int i = 0; i < nums.length; i++) {
            if (sum > 0) {
                sum += nums[i];
            } else {
                sum = nums[i];
            }
            ans = Math.max(sum, ans);
        }
        return ans;
    }
}

```

#### 分治

分治法的思路是[这样的](https://leetcode-cn.com/problems/maximum-subarray/solution/zheng-li-yi-xia-kan-de-dong-de-da-an-by-lizhiqiang/)，其实也是分类讨论。

连续子序列的最大和主要由这三部分子区间里元素的最大和得到：

第 1 部分：子区间 [left, mid]；
第 2 部分：子区间 [mid + 1, right]；
第 3 部分：包含子区间[mid , mid + 1]的子区间，即 `nums[mid]` 与``nums[mid + 1]`一定会被选取。
对它们三者求最大值即可。

![image.png](https://pic.leetcode-cn.com/06b132ae309c010ea12ebc8516df52827fd720735fb6eec3394e1a7cf1fadde4-image.png) 

```java
class Solution {
    public int maxSubArray(int[] nums) {
        if (nums.length == 1)
            return nums[0];
        return maxSubArrayDivideWithBorder(nums, 0, nums.length - 1);
    }

    private int maxSubArrayDivideWithBorder(int[] nums, int start, int end) {
        // 只有一个元素，也就是递归的结束情况
        if (start == end)
            return nums[start];
        int mid = start + (end - start) / 2;
        // 递归计算左侧子序列最大值
        int leftMaxVal = maxSubArrayDivideWithBorder(nums, start, mid);
        // 递归计算右侧子序列最大值
        int rightMaxVal = maxSubArrayDivideWithBorder(nums, mid + 1, end);

        int leftCrossMaxVal = Integer.MIN_VALUE, rightCrossMaxVal = Integer.MIN_VALUE;
        int leftSum = 0, rightSum = 0;
        // 计算包含左侧子序列最后一个元素的子序列最大值，从右往左一个个加
        for (int i = mid; i >= 0; i--) {
            leftSum += nums[i];
            leftCrossMaxVal = Math.max(leftCrossMaxVal, leftSum);
        }
         // 计算包含右侧子序列第一个元素的子序列最大值，从左往右一个个加
        for (int j = mid + 1; j <= end; j++) {
            rightSum += nums[j];
            rightCrossMaxVal = Math.max(rightCrossMaxVal, rightSum);
        }
        return Math.max(leftCrossMaxVal + rightCrossMaxVal, Math.max(leftMaxVal, rightMaxVal));
    }
}

```

- 时间复杂度：`O(NlogN)`，这里递归的深度是对数级别的，每一层需要遍历一遍数组（或者数组的一半、四分之一）；
- 空间复杂度：`O(logN)`，需要常数个变量用于选取最大值，需要使用的空间取决于递归栈的深度。

## BFS

### 二叉树的最小高度

```java
/**
 * Definition for a binary tree node. public class TreeNode { int val; TreeNode
 * left; TreeNode right; TreeNode() {} TreeNode(int val) { this.val = val; }
 * TreeNode(int val, TreeNode left, TreeNode right) { this.val = val; this.left
 * = left; this.right = right; } }
 */
class Solution {
    public int minDepth(TreeNode root) {
        if (root == null)
            return 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int depth = 1;
        while (!queue.isEmpty()) {
            int sz = queue.size();
            for (int i = 0; i < sz; i++) {
                TreeNode node = queue.remove();
                //到达终点
                if (node.left == null && node.right == null)
                    return depth;
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
            }
            depth++;
        }
        return depth;
    }
}

```

### 752 打开转盘锁

```java
你有一个带有四个圆形拨轮的转盘锁。每个拨轮都有10个数字： '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' 。每个拨轮可以自由旋转：例如把 '9' 变为  '0'，'0' 变为 '9' 。每次旋转都只能旋转一个拨轮的一位数字。

锁的初始数字为 '0000' ，一个代表四个拨轮的数字的字符串。

列表 deadends 包含了一组死亡数字，一旦拨轮的数字和列表里的任何一个元素相同，这个锁将会被永久锁定，无法再被旋转。

字符串 target 代表可以解锁的数字，你需要给出最小的旋转次数，如果无论如何不能解锁，返回 -1。

 

示例 1:

输入：deadends = ["0201","0101","0102","1212","2002"], target = "0202"
输出：6
解释：
可能的移动序列为 "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202"。
注意 "0000" -> "0001" -> "0002" -> "0102" -> "0202" 这样的序列是不能解锁的，
因为当拨动到 "0102" 时这个锁就会被锁定。

示例 2:

输入: deadends = ["8888"], target = "0009"
输出：1
解释：
把最后一位反向旋转一次即可 "0000" -> "0009"。

示例 3:

输入: deadends = ["8887","8889","8878","8898","8788","8988","7888","9888"], target = "8888"
输出：-1
解释：
无法旋转到目标数字且不被锁定。

```

每一次转动8种可能，四个数字向上加一，四个数字向下减一，直接套用BFS框架

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        Set<String> deads = new HashSet<>();
        //防止走回头路
        Set<String> visited = new HashSet<>();
        //记录死亡数字
        for (String string : deadends) {
            deads.add(string);
        }
        Queue<String> q = new LinkedList<>();
        //转动次数
        int step = 0;
        q.offer("0000");
        visited.add("0000");
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                String s = q.poll();
                //死亡数字
                if (deads.contains(s))
                    continue;
                if (s.equals(target))
                    return step;
                
                for (int j = 0; j < 4; j++) {
                    String up = plusOne(s, j);
                    if (!visited.contains(up)) {
                        q.offer(up);
                        visited.add(up);
                    }
                    String down = minusOne(s, j);
                    if (!visited.contains(down)) {
                        q.offer(down);
                        visited.add(down);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    private String plusOne(String s, int j) {
        char[] ch = s.toCharArray();
        if (ch[j] == '9')
            ch[j] = '0';
        else
            ch[j] += 1;
        return new String(ch);
    }

    private String minusOne(String s, int j) {
        char[] ch = s.toCharArray();
        if (ch[j] == '0')
            ch[j] = '9';
        else
            ch[j] -= 1;
        return new String(ch);
    }
}

```

也可以只用`visited`记录死亡数字，但`visited.add(s)`要放在加一减一的循环外面，不然新加的字符串会被`if(visited.contains(s))`过滤掉

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        // Set<String> deads = new HashSet<>();
        Set<String> visited = new HashSet<>();
        for (String string : deadends) {
            visited.add(string);
        }
        Queue<String> q = new LinkedList<>();
        int step = 0;
        q.offer("0000");
        // visited.add("0000");
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                String s = q.poll();
                if (visited.contains(s))
                    continue;
                if (s.equals(target))
                    return step;
                visited.add(s);
                for (int j = 0; j < 4; j++) {
                    String up = plusOne(s, j);
                    if (!visited.contains(up)) {
                        q.offer(up);
                        // visited.add(up);
                    }
                    String down = minusOne(s, j);
                    if (!visited.contains(down)) {
                        q.offer(down);
                        // visited.add(down);
                    }
                }
            }
            step++;
        }
        return -1;
    }

    private String plusOne(String s, int j) {
        char[] ch = s.toCharArray();
        if (ch[j] == '9')
            ch[j] = '0';
        else
            ch[j] += 1;
        return new String(ch);
    }

    private String minusOne(String s, int j) {
        char[] ch = s.toCharArray();
        if (ch[j] == '0')
            ch[j] = '9';
        else
            ch[j] -= 1;
        return new String(ch);
    }
}

```

#### 双向BFS

参考`labuladong`

 传统的 BFS 框架就是从起点开始向四周扩散，遇到终点时停止；而双向 BFS 则是从起点和终点同时开始扩散，当两 边有交集的时候停⽌。  

![1618148554681](C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618148554681.png)

![1618148572670](C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618148572670.png)

```java
class Solution {
    public int openLock(String[] deadends, String target) {
        // Set<String> deads = new HashSet<>();
        Set<String> deads = new HashSet<>();
        for (String string : deadends) {
            deads.add(string);
        }
        // ⽤集合不⽤队列，可以快速判断元素是否存在
        Set<String> visited = new HashSet<>();
        Set<String> q1 = new HashSet<>();
        Set<String> q2 = new HashSet<>();
        int step = 0;
        q1.add("0000");
        q2.add(target);
        while (!q1.isEmpty() && !q2.isEmpty()) {
            // 哈希集合在遍历的过程中不能修改，⽤ temp 存储扩散结果
            Set<String> temp = new HashSet<>();
            for (String s : q1) {
                if (deads.contains(s))
                    continue;
                if (q2.contains(s))
                    return step;
                visited.add(s);
                for (int j = 0; j < 4; j++) {
                    String up = plusOne(s, j);
                    if (!visited.contains(up))
                        temp.add(up);
                    String down = minusOne(s, j);
                    if (!visited.contains(down))
                        temp.add(down);
                }
            }
            step++;
            // temp 相当于 q1
            // 这⾥交换 q1 q2，下⼀轮 while 就是扩散 q2
            q1 = q2;
            q2 = temp;
        }

        return -1;
    }

    private String plusOne(String s, int j) {
        char[] ch = s.toCharArray();
        if (ch[j] == '9')
            ch[j] = '0';
        else
            ch[j] += 1;
        return new String(ch);
    }

    private String minusOne(String s, int j) {
        char[] ch = s.toCharArray();
        if (ch[j] == '0')
            ch[j] = '9';
        else
            ch[j] -= 1;
        return new String(ch);
    }
}

```

 其实双向 BFS 还有⼀个优化，就是在 while 循环开始时做⼀个判断：  

```java
 while (!q1.isEmpty() && !q2.isEmpty()) {
    //交换放到上面，从长度更少的一方开始
     if (q1.size() > q2.size()) {
         Set<String> temp = new HashSet<>();
         temp = q1;
         q1 = q2;
         q2 = temp;
     }
     Set<String> temp = new HashSet<>();
     //...
 }

```

 为什么这是⼀个优化呢？ 因为按照 BFS 的逻辑，队列（集合）中的元素越多，扩散之后新的队列 （集合）中的元素就越多；在双向 BFS 算法中，如果我们每次都选择⼀个 较⼩的集合进⾏扩散，那么占⽤的空间增⻓速度就会慢⼀些，效率就会⾼⼀ 些。 

### 1091 二进制矩阵中的最短路径

 题目描述：0 表示可以经过某个位置，求解从左上角到右下角的最短路径长度。 

```java
class Solution {
    public int shortestPathBinaryMatrix(int[][] grid) {
        int n = grid.length;
        if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1)
            return -1;
        if (n == 1)
            return 1;
        Queue<int[]> q = new LinkedList<>();
        q.add(new int[] { 0, 0 });
        int step = 1;
        //位置走向，八个方向
        int[][] dir = { { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, { 1, 1 } };
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int[] pos = q.poll();
                //到达终点
                if (pos[0] == n - 1 && pos[1] == n - 1)
                    return step;

                for (int[] direction : dir) {
                    int pos_0 = pos[0] + direction[0];
                    int pos_1 = pos[1] + direction[1];
                    //出边界
                    if (pos_0 < 0 || pos_1 < 0 || pos_0 > n - 1 || pos_1 > n - 1)
                        continue;
                    //访问过的或不能走的,访问过说明已经走过，step更小，没必要再走了
                    if (grid[pos_0][pos_1] == 1)
                        continue;
                    // 标记访问
                    grid[pos_0][pos_1] = 1;
                    
                    int[] newPos = new int[] { pos_0, pos_1 };
                    q.offer(newPos);

                }
            }
            step++;
        }
        return -1;
    }
}

```

### 279 组成整数的最小平方数数量

可以将每个整数看成图中的一个节点，如果两个整数之差为一个平方数，那么这两个整数所在的节点就有一条边。

要求解最小的平方数数量，就是求解**从节点 n 到节点 0 的最短路径**。

```java
class Solution {
    public int numSquares(int n) {
        List<Integer> squares = generateSquares(n);
        Queue<Integer> q = new LinkedList<>();
        q.add(n);
        boolean[] marked = new boolean[n + 1];
        marked[n] = true;
        int level = 0;
        while (!q.isEmpty()) {
            int sz = q.size();
            //防止到达终点后提前返回level，放前面
            level++;
            while (sz-- > 0) {
                int cur = q.poll();
                for (int square : squares) {
                    int next = cur - square;
                    if (next < 0)
                        break;
                    if (next == 0)
                        return level;
                    if (marked[next] == true)
                        continue;
                    q.add(next);
                    marked[next] = true;
                }
            }
        }
        return n;
    }
    //1 4 9...
    private List<Integer> generateSquares(int n) {
        List<Integer> squares = new ArrayList<>();
        int i = 1, increment = 3;
        while (i <= n) {
            squares.add(i);
            i += increment;
            increment += 2;
        }
        return squares;
    }
}

```

#### 动态规划

```java
class Solution {
    public int numSquares(int n) {
        List<Integer> squares = generateSquares(n);
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        // int min = Integer.MAX_VALUE;
        for (int i = 1; i <= n; i++) {
            for (int square : squares) {
                if (i - square < 0)
                    break;
                dp[i] = Math.min(dp[i], dp[i - square] + 1);
            }

        }
        return dp[n];
    }
    //1 4 9...
    private List<Integer> generateSquares(int n) {
        List<Integer> squares = new ArrayList<>();
        int i = 1, increment = 3;
        while (i <= n) {
            squares.add(i);
            i += increment;
            increment += 2;
        }
        return squares;
    }
}

```

### 127 最短单词路径

```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        if (!words.contains(endWord))
            return 0;
        Queue<String> queue = new LinkedList<>();
        boolean[] visited = new boolean[wordList.size()];
        // -1 if not existed
        queue.add(beginWord);
        if (wordList.indexOf(beginWord) != -1) {
            visited[wordList.indexOf(beginWord)] = true;
        }
        int cnt = 0;
        while (!queue.isEmpty()) {
            int sz = queue.size();
            cnt++;
            while (sz-- > 0) {
                String cur = queue.poll();
                // String cur = wordList.get(idx);
                for (int i = 0; i < wordList.size(); i++) {
                    String s = wordList.get(i);
                    if (!canConvert(cur, s))
                        continue;
                    if (visited[i])
                        continue;
                    if (s.equals(endWord))
                        return cnt + 1;
                    queue.add(s);
                    visited[i] = true;
                }
            }
        }
        return 0;
    }

    private boolean canConvert(String s, String t) {
        int cnt = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) != t.charAt(i))
                cnt++;
            if (cnt > 1)
                return false;
        }
        return true;
    }
}

```

#### 双向BFS

```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        if (!words.contains(endWord))
            return 0;
        // words.add(beginWord);
        Queue<String> q1 = new LinkedList<>();
        Queue<String> q2 = new LinkedList<>();
        boolean[] visited1 = new boolean[wordList.size()];
        boolean[] visited2 = new boolean[wordList.size()];
        // -1 if not exsisted
        q1.add(beginWord);
        q2.add(endWord);
        if (wordList.indexOf(beginWord) != -1) {
            visited1[wordList.indexOf(beginWord)] = true;
        }
        visited2[wordList.indexOf(endWord)] = true;
        int cnt = 0;
        while (!q1.isEmpty()) {
            if (q1.size() > q2.size()) {
                Queue<String> temp = new LinkedList<>();
                temp = q1;
                q1 = q2;
                q2 = temp;
                boolean[] v = new boolean[wordList.size()];
                v = visited1;
                visited1 = visited2;
                visited2 = v;
            }
            int sz = q1.size();
            cnt++;
            while (sz-- > 0) {
                String cur = q1.poll();
                // String cur = wordList.get(idx);
                for (int i = 0; i < wordList.size(); i++) {
                    String s = wordList.get(i);
                    if (!canConvert(cur, s))
                        continue;
                    if (visited1[i])
                        continue;
                    //相遇
                    if (visited2[wordList.indexOf(s)] == true)
                        return cnt + 1;
                    q1.add(s);
                    visited1[i] = true;
                }
            }
        }
        return 0;
    }

    private boolean canConvert(String s, String t) {
        int cnt = 0;
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) != t.charAt(i))
                cnt++;
            if (cnt > 1)
                return false;
        }
        return true;
    }
}

```

**进一步优化**

单词转换判断的优化 (20ms)
判断当前单词可以转换成哪些候选单词（未访问的单词），有两种思路：

**思路1**：遍历所有候选单词，判断当前单词是否可以转换成这个候选单词。判断的过程也就是前面的`canConvert`方法，逐个对比单词的字符。

**思路2**：因为单词是由 a~z 这有限数量的字符组成的，可以遍历当前单词能转换成的所有单词，判断其是否包含在候选单词中。候选单词用 `HashSet` 保存，**可以大大提高判断包含关系的性能，比用 `boolean` 数组更快**。

当单词总数量庞大的时候，之前代码用到的思路1耗时就会很长。而当单词的字符串数量、单词长度很大时，思路 2 耗时就会更长。实际情况下，一般单词不会很长，字符也是固定的 26 个小写字母，因此思路2的性能会好很多。

于是进一步优化代码，思路1改为思路 2，性能提升很明显，耗时减少到了 20ms。

```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        if (!words.contains(endWord))
            return 0;
        // words.add(beginWord);
        Queue<String> q1 = new LinkedList<>();
        Queue<String> q2 = new LinkedList<>();
        Set<String> visited1 = new HashSet<>();
        Set<String> visited2 = new HashSet<>();
        
        // -1 if not exsisted
        q1.add(beginWord);
        q2.add(endWord);
        visited1.add(beginWord);
        visited2.add(endWord);
        int cnt = 0;
        while (!q1.isEmpty()) {
            if (q1.size() > q2.size()) {
                Queue<String> temp = new LinkedList<>();
                temp = q1;
                q1 = q2;
                q2 = temp;
                Set<String> v = new HashSet<>();
                v = visited1;
                visited1 = visited2;
                visited2 = v;
            }
            int sz = q1.size();
            cnt++;
            while (sz-- > 0) {
                String cur = q1.poll();
                char[] chars = cur.toCharArray();
                // String cur = wordList.get(idx);
                for (int i = 0; i < cur.length(); i++) {
                    char c0 = chars[i];
                    for (char c = 'a'; c <= 'z'; c++) {
                        chars[i] = c;
                        String str = new String(chars);
                        // if (!canConvert(cur, s))
                        // continue;
                        if (!words.contains(str))
                            continue;
                        // int index = wordList.indexOf(str);
                        //访问过了
                        if (visited1.contains(str))
                            continue;
                        //相遇
                        if (visited2.contains(str))
                            return cnt + 1;
                        q1.add(str);
                        visited1.add(str);
                    }
                    chars[i] = c0;
                }
            }
        }
        return 0;
    }
}

```

## 并查集

**参考`labuladong`公众号**

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618968326059.png" alt="1618968326059" style="zoom:50%;" />

![路径压缩，高度保持常数](C:\Users\32332\Desktop\计算机知识点总结\力扣相关图片\微信图片_20210421092547.gif)

有了路径压缩，其实可以不用统计`size`，个人理解

```java
public class Solution {
    
    public static void main(String[] args) {
        Solution s = new Solution();
       
        UF uf = s.new UF(4);
        uf.union(1, 2);
        uf.union(3, 0);
        uf.union(1, 0);
        System.out.println(uf.isConnected(1, 2));
    }

    class UF {
        int[] parent;
        int[] size;
        int count;
        public UF(int n) {
            parent = new int[n];
            size = new int[n];
            //连通分量个数
            count = n;
            for (int i = 0; i < n; i++) {
                parent[i] = i;
                size[i] = 1;
            }
        }

        private int find(int x) {
            while (parent[x] != x) {
                //路径压缩，并且保持高度不超过3（union时可能会超过3）
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }
        //连通
        private void union(int x, int y) {
            int rootx = find(x);
            int rooty = find(y);
            //小树接到大树下面
            if (size[rootx] < size[rooty]) {
                parent[rootx] = rooty;
                size[rooty] += size[rootx];
            } else {
                parent[rooty] = rootx;
                size[rootx] += size[rooty];
            }
            count--;
        }

        private boolean isConnected(int x, int y) {
            int rootx = find(x);
            int rooty = find(y);
            return rootx == rooty;
        }
    }    
}

```

## 哈夫曼树

**定义**：叶子结点和权重确定的情况下，带权路径长度（**WPL**）最小的二叉树，也被称为**最优二叉树**

**即**权重小的叶子结点尽量离根结点远

优先队列保存所有权重叶子结点，按从小到大顺序排列，选择当前权值最小的两个结点，生成新的父结点，把父结点加入优先队列

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1619765580040.png" alt="1619765580040" style="zoom:50%;" />

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1619765600730.png" alt="1619765600730" style="zoom:50%;" />

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1619765711698.png" alt="1619765711698" style="zoom:50%;" />



```java
public class HuffmanTree {
    public static void main(String[] args) {
        int[] weights = new int[] { 2, 3, 7, 9, 18, 25 };
        HuffmanTree hTree = new HuffmanTree();
        hTree.createHuffman(weights);
        hTree.output(hTree.root);
    }
    private Node root;
    private Node[] nodes;

    public void createHuffman(int[] weights) {
        Queue<Node> nodeQueue = new PriorityQueue<>();
        nodes = new Node[weights.length];
        for (int i = 0; i < weights.length; i++) {
            nodes[i] = new Node(weights[i]);
            nodeQueue.add(nodes[i]);
        }
        while (nodeQueue.size() > 1) {
            Node lChild = nodeQueue.poll();
            Node rChild = nodeQueue.poll();
            Node parent = new Node(lChild.weight + rChild.weight, lChild, rChild);
            nodeQueue.add(parent);
        }
        root = nodeQueue.poll();
    }
	//前序遍历
    public void output(Node head) {
        if (head == null)
            return;
        System.out.println(head.weight);
        output(head.lChild);
        output(head.rChild);
    }
    class Node implements Comparable<Node>{
        int weight;
        Node lChild;
        Node rChild;

        public Node(int weight) {
            this.weight = weight;
        }

        public Node(int weight, Node left, Node right) {
            this.weight = weight;
            this.lChild = left;
            this.rChild = right;
        }
        
        public int compareTo(Node o) {
            return this.weight - o.weight;
        }
    }
}

```

### 哈夫曼编码

权重*路径长度 **等价** 字符出现频次 * 编码长度

## DFS

###  695 岛屿的最大面积

```java
class Solution {

    // private int[][] dir = new int[][] { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 }
    // };

    public int maxAreaOfIsland(int[][] grid) {

        int maxRow = grid.length;
        int maxCol = grid[0].length;
        int maxArea = 0;
        for (int i = 0; i < maxRow; i++) {
            for (int j = 0; j < maxCol; j++) {
                maxArea = Math.max(maxArea, dfs(grid, i, j));
            }
        }
        return maxArea;
    }

    private int dfs(int[][] grid, int row, int col) {
        if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length || grid[row][col] == 0)
            return 0;
        grid[row][col] = 0;
        return 1 + dfs(grid, row + 1, col) + dfs(grid, row - 1, col) + dfs(grid, row, col + 1)
                + dfs(grid, row, col - 1);
    }
}

```

### 547 省份数量

无向图连通分量

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618926718851.png" alt="1618926718851" style="zoom:80%;" />

```java
class Solution {
    public int findCircleNum(int[][] isConnected) {
        boolean[] visited = new boolean[isConnected.length];
        int res = 0;
        for (int i = 0; i < isConnected.length; i++) {
            if (!visited[i]) {
                dfs(isConnected, i, visited);
                res++;
            }
        }
        return res;
    }

    private void dfs(int[][] nums, int i, boolean[] visited) {
        visited[i] = true;
        for (int j = 0; j < nums.length; j++) {
            if (nums[i][j] == 1 && !visited[j]) {
                dfs(nums, j, visited);
            }
        }
    }
}

```

### 130 填充封闭区域

被`X`包围的`O`用`X`填充，外围的不算

**思路**：先将与外围`O`连通的`O`找出来，用`T`填充，再找被`X`包围的`O`，最后把`T`变回去

```java
class Solution {
    public void solve(char[][] board) {
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; i++) {
            dfs(board, i, 0);
            dfs(board, i, n - 1);
        }
        for (int j = 0; j < n; j++) {
            dfs(board, 0, j);
            dfs(board, m - 1, j);
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
                if (board[i][j] == 'T')
                    board[i][j] = 'O';
            }   
        }
    }

    private void dfs(char[][] board, int i, int j) {
        if (i < 0 || i > board.length - 1 || j < 0 || j > board[0].length - 1 || board[i][j] != 'O')
            return;
        board[i][j] = 'T';
        dfs(board, i - 1, j);
        dfs(board, i + 1, j);
        dfs(board, i, j - 1);
        dfs(board, i, j + 1);
    }
}

```

#### 并查集

```java
class Solution {
    class UF {
        int[] parent;

        public UF(int n) {
            parent = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }

        private int find(int x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        private void union(int x, int y) {
            int root_x = find(x);
            int root_y = find(y);
            if (root_x == root_y)
                return;
            parent[root_x] = root_y;
        }

        private boolean isConnected(int x, int y) {
            int root_x = find(x);
            int root_y = find(y);
            return root_x == root_y;
        }
    }

    public void solve(char[][] board) {
        int[][] dir = new int[][] { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
        int m = board.length;
        int n = board[0].length;
        int[] parent = new int[m * n + 1];
        //留一个位置给dummy
        int dummy = parent[m * n];
        UF uf = new UF(m * n);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (board[i][j] == 'O') {
                    if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                        //边界的'O'与dummy连通
                        uf.union(i * n + j, dummy);
                    } else {
                        for (int[] pos : dir) {
                            int x = i + pos[0];
                            int y = j + pos[1];
                            if (!inArea(x, y, m, n))
                                continue;
                            if (board[x][y] == 'O') {
                                uf.union(x * n + y, i * n + j);
                            }
                        }
                    }
                }
            }
        }
        //不与dummy连通的'O'是被包围的
        for (int i = 1; i < m - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                if (!uf.isConnected(dummy, i * n + j))
                    board[i][j] = 'X';
            }
        }
    }

    private boolean inArea(int x, int y, int row, int col) {
        return (x >= 0 && x < row && y >= 0 && y < col);
    }

}

```



## 跳跃游戏

### 55 跳跃游戏I

```java
给定一个非负整数数组 nums ，你最初位于数组的 第一个下标 。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标。

示例 1：

输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
    
示例 2：

输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。

```

```java
class Solution {
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int far = 0;
        for (int i = 0; i < n - 1; i++) {
            //i + nums[i] 是当前位置跳的最远距离
            far = Math.max(far, i + nums[i]);
            //碰到 0，卡住跳不动了
            if (far <= i)
                return false;
        }
        return far >= n - 1;
    }
}

```

### 45 跳跃游戏II

```java
给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

示例:

输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

```

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618801961340.png" alt="1618801961340" style="zoom:70%;" />

```java
class Solution {
    public int jump(int[] nums) {
        int jumps = 0;
        int farthest = 0;
        int end = 0;
        //注意边界
        for (int i = 0; i < nums.length - 1; i++) {
            farthest = Math.max(farthest, i + nums[i]);
            //更新边界，从 0 开始跳
            if (end == i) {
                jumps++;
                end = farthest;
            }
        }
        return jumps;
     }
}

```

### 1306 跳跃游戏III

#### DFS

```java
class Solution {
    private int flag = 0;

    public boolean canReach(int[] arr, int start) {
       int[] visited = new int[arr.length];
 
       dfs(arr, start, visited);
       return (flag == 1);
    }
 
    private void dfs(int[] nums, int i, int[] visited) {
       if (visited[i] == 1)
          return;
       if (nums[i] == 0) {
          flag = 1;
       }
       visited[i] = 1;
       if (i + nums[i] <= nums.length - 1)
          dfs(nums, i + nums[i], visited);
       if (i - nums[i] >= 0)
          dfs(nums, i - nums[i], visited);
    }
}

```

### 1345 跳跃游戏IV

典型BFS

```java
class Solution {
    public int minJumps(int[] arr) {
        int len = arr.length;
        if (len == 1)
            return 0;
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int i = 0; i < arr.length; i++) {
            Set<Integer> set = map.getOrDefault(arr[i], new HashSet<Integer>());
            set.add(i);
            map.put(arr[i], set);
        }
        Queue<Integer> q = new LinkedList<>();
        // int[] visited = new int[len];
        int[] dis = new int[len];
        q.add(0);
        dis[0] = 1;
        while (!q.isEmpty()) {
            // int sz = q.size();
            int size = q.size();
            while (size-- > 0) {

                int i = q.remove();
                if (i == len - 1)
                    break;
                if (i - 1 > 0 && dis[i - 1] == 0) {
                    q.add(i - 1);
                    dis[i - 1] = dis[i] + 1;
                }
                if (i + 1 < len && dis[i + 1] == 0) {
                    q.add(i + 1);
                    dis[i + 1] = dis[i] + 1;
                }
                if (map.containsKey(arr[i])) {
                    for (int j : map.get(arr[i])) {
                        if (dis[j] == 0) {
                            q.add(j);
                            dis[j] = dis[i] + 1;
                        }
                    }
                    //不加这句通不过
                    map.remove(arr[i]);
                }

            }
        }
        return dis[len - 1] - 1;
    }
}

```



####  BFS

```JAVA
class Solution {
    // private int flag = 0;

    public boolean canReach(int[] arr, int start) {
        int[] visited = new int[arr.length];
        Queue<Integer> q = new LinkedList<>();
        q.add(start);
        // visited[start] = 1;
        // boolean flag = false;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; i++) {
                int cur = q.poll();

                if (cur < 0 || cur > arr.length - 1)
                    continue;
                if (visited[cur] == 1)
                    continue;
                if (arr[cur] == 0)
                    return true;
                visited[cur] = 1;
                q.add(cur + arr[cur]);
                q.add(cur - arr[cur]);
            }
        }
        return false;
    }    
}

```

## 回溯

### 22 括号生成

生成n对括号的所有配对组合

```java
class Solution {
    public List<String> generateParenthesis(int n) {
        List<String> res = new LinkedList<>();
        String s = "";
        track(res, s, 0, 0, n);
        return res;
    }

    private void track(List<String> res, String s, int left, int right,int n) {
        if (left < right)
            return;
        if (left > n || right > n)
            return;
        if (left == n && right == n)
            res.add(s);
        s += "(";
        track(res, s, left + 1, right, n);
        s = s.substring(0, s.length() - 1);
        s += ")";
        track(res, s, left, right + 1, n);
        s = s.substring(0, s.length() - 1);
        
    }
}

```

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0: 
            return []

        self.s = ""
        self.res = []
        self.backtrack(self.res, self.s, n, n)
        return self.res
        
    def backtrack(self, res, s, left, right):
        if left < 0 or right < 0:
            return 
        if right < left:
            return
        if left == 0 and right == 0:
            res.append(s)
            return
        
        self.backtrack(res, s + '(', left - 1, right)   
        self.backtrack(res, s + ')', left, right - 1)

```



### 47 全排列II

有重复数字

```
如 nums = [1 1 2],全排列为[1 1 2] [1 2 1] [2 1 1]
 2 1 1只需要遍历一次，第一次遍历的时候 visited[0] = true,visited[1] = true,后面回溯都变为false
 第二次遍历到nums[1]的时候，nums[0]为false,跳过本轮循环

```

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1619749984425.png" alt="1619749984425" style="zoom:60%;" />

```java
class Solution {
    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res  = new ArrayList<>();
        ArrayList<Integer> ans = new ArrayList<>();
        if(nums.length == 0) return res;
        //先排序
        Arrays.sort(nums);
        boolean[] visited = new boolean[nums.length];
        dfs(nums,visited,res,ans);
        return res;
    }
    private void dfs(int[] nums,boolean[] visited,List<List<Integer>> res,List<Integer> ans) {
        if(ans.size() == nums.length){
            res.add(new ArrayList<>(ans));
            return;
        }
        for(int i = 0; i < nums.length; i++){
            if(visited[i]) continue;
            if(i > 0 && nums[i - 1] == nums[i] && !visited[i - 1]) continue;
            ans.add(nums[i]);
            visited[i] = true;
            dfs(nums,visited,res,ans);
            visited[i] = false;
            ans.remove(ans.size() - 1);
        }
    }
}

```



### 17 数字键盘组合

```java
class Solution {
    public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        if (digits.length() == 0)
            return res;
        StringBuilder sb = new StringBuilder();
        dfs(digits, sb, res);
        return res;
    }

    private String[] letters = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
    private void dfs(String digits, StringBuilder sb, List<String> res) {
        if (sb.length() == digits.length()) {
            res.add(sb.toString());
            //结束本轮递归，不然到下面digits会索引出界
            return;
        }
        String letter = letters[digits.charAt(sb.length()) - '0'];
        for (char c : letter.toCharArray()) {
            sb.append(c);
            dfs(digits, sb, res);
            sb.deleteCharAt(sb.length() - 1);
        }
    }
}

```

#### 法二:队列

```java
digits = "23"为例
a b c -> ad ae af;bd be bf;cd ce cf

```

```java
class Solution {
    public List<String> letterCombinations(String digits) {
        List<String> res = new LinkedList<>();
        if (digits.length() == 0)
            return res;
        String[] letters = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };

        res.add("");
        for (int i = 0; i < digits.length(); i++) {
            String letter = letters[digits.charAt(i) - '0'];
            int size = res.size();
            while(size-- > 0){
                String s = res.remove(0);
                for (int j = 0; j < letter.length(); j++) {
                    res.add(s + letter.charAt(j));
                }
            }
        }
        return res;
    }
}

```

### 37 数独

9 * 9矩阵，往空格里填数字，所在行所在列不能出现同样的数字，3 * 3小方框里同样不能出现相同的数字

```java
class Solution {
    public void solveSudoku(char[][] board) {
        isValidMatrix(board, 0, 0);
    }
    private boolean isValidMatrix(char[][] board, int r, int c) {
        if (c == 9)
            return isValidMatrix(board, r + 1, 0);
        if (r == 9)
            return true;
        if (board[r][c] != '.')
            return isValidMatrix(board, r, c + 1);
        for (char ch = '1'; ch <= '9'; ch++) {
            if (!isValid(board, r, c, ch))
                continue;
            board[r][c] = ch;
            //找到一个可行解，立马返回
            if (isValidMatrix(board, r, c + 1)) {
                return true;
            }
            board[r][c] = '.';
        }
        return false;
    }

    private boolean isValid(char[][] board, int r, int c, char ch) {
        for (int i = 0; i < 9; i++) {
            if (board[r][i] == ch || board[i][c] == ch)
                return false;
            if (board[r/3*3 + i/3][c/3*3 + i%3] == ch)
                return false;
        }
        return true;
    }
}

```

### N皇后

 在 n*n 的矩阵中摆放 n 个皇后，并且每个皇后不能在同一行，同一列，同一对角线上，求所有的 n 皇后的解。 

```java
class Solution {
    public List<List<String>> solveNQueens(int n) {
        String[][] board = new String[n][n];
        for (String[] str : board) {
            Arrays.fill(str, ".");
        }
        List<List<String>> res = new LinkedList<>();
        solve(board, 0, res);
        return res;
    }

    private void solve(String[][] board, int row, List<List<String>> res) {
        int n = board.length;
        if (row == n) {
            List<String> ans = new ArrayList<>();
            for (String[] str : board) {
                ans.add(String.join("", str));
            }
            res.add(ans);
            return;
        }
        for (int col = 0; col < n; col++) {
            if (!isValid(board, row, col))
                continue;
            board[row][col] = "Q";
            solve(board, row + 1, res);
            board[row][col] = ".";
        }
    }

    private boolean isValid(String[][] board, int r, int c) {
        for (int i = 0; i < board.length; i++) {
            if (board[r][i] == "Q" || board[i][c] == "Q")
                return false;
        }
        for (int i = r - 1, j = c - 1; i >= 0 && j >= 0; i--, j--) {
            if (board[i][j] == "Q")
                return false;
        }
        for (int i = r - 1, j = c + 1; i >= 0 && j < board.length; i--, j++) {
            if (board[i][j] == "Q")
                return false;
        }
        return true;
    }
}

```



### 79 单词搜索

```java
class Solution {
    private boolean res;
    private int rows;
    private int cols;
    private int len;
    private char[][] board;
    private char[] charArray;
    private boolean[][] visited;

    public boolean exist(char[][] board, String word) {
        this.rows = board.length;
        this.cols = board[0].length;
        this.len = word.length();
        if (rows == 0 || rows * cols < len)
            return false;
        this.charArray = word.toCharArray();
        this.board = board;
        this.visited = new boolean[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (board[i][j] == charArray[0] && dfs(i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean dfs(int i, int j, int start) {
        if (!inArea(i, j) || board[i][j] != charArray[start] || visited[i][j])
            return false;
        
        if (start == len - 1) {
            return true;
        }
        
        visited[i][j] = true;
        
        res = dfs(i - 1, j , start + 1) || dfs(i + 1, j, start + 1)
                || dfs(i, j - 1, start + 1) || dfs(i, j + 1, start + 1);

        visited[i][j] = false;
        return res;
    }

    private boolean inArea(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            return false;
        }
        return true;
    }
}

```



### 93 合法`ip`

- 暴力，每个字段判断是否合法

```java
class Solution {
    public List<String> restoreIpAddresses(String s) {
        List<String> res = new LinkedList<>();
        if (s.length() > 12 || s.length() < 4)
            return res;
        int n = s.length();
        String s1, s2, s3, s4;
        //字段长度不超过3，且剩下不少于3位
        for (int i1 = 1; i1 <= 3 && n - i1 >= 3; i1++) {
            //左闭右开
            s1 = s.substring(0, i1);
            if (!isValid(s1))
                continue;
            //字段长度不超过3，且剩下不少于2位
            for (int i2 = i1 + 1, j2 = 0; j2 < 3 && n - i2 >= 2; i2++, j2++) {
                s2 = s.substring(i1, i2);
                if (!isValid(s2))
                    continue;
                for (int i3 = i2 + 1, j3 = 0; j3 < 3 && n - i3 >= 1; i3++, j3++) {
                    s3 = s.substring(i2, i3);
                    s4 = s.substring(i3);
                    if (!isValid(s3) || !isValid(s4))
                        continue;
                    s4 = s1 + '.'+ s2 + '.' + s3 + '.' + s4;
                    res.add(s4);
                }
            }
        }
        return res;
    }

    private boolean isValid(String s) {
        if (s.length() > 1 && s.charAt(0) == '0')
            return false;
        int cur = Integer.parseInt(s);
        if (cur < 0 || cur > 255)
            return false;
        return true;
    }
}

```

####  回溯

```java
class Solution {
    public List<String> restoreIpAddresses(String s) {
        LinkedList<String> res = new LinkedList<>();
        LinkedList<String> list = new LinkedList<>();
        if (s.length() > 12 || s.length() < 4)
            return res;
        // int n = s.length();
        dfs(s, 1, 0, res, list);
        return res;
    }

    private void dfs(String s, int step, int left, LinkedList<String> res, LinkedList<String> list) {
        //最多四轮，由于step初始化为1，第四轮时step=4,接着进入递归，此时step = 5,结束递归
        if (step == 5) {
            res.add(String.join(".", list));
            return;
        }
        //每一个字段长度不超过3，right最多left后2位
        for (int right = left; right < left + 3 && right < s.length(); right++) {
            //左闭右开
            String str = s.substring(left, right + 1);
            //类似"01"字段不被允许
            if (str.length() > 1 && str.charAt(0) == '0')
                continue;
            //如12位长度，第一个字段只能是3位，第一轮step=1,第一个字段长度1和2都不符合，后面位数超了
            //再比如，第一轮step=1,但后面只剩2位了，也不符合
            if (s.length() - 1 - right < 4 - step || s.length() - 1 - right > 3 * (4 - step))
                continue;
            //数值范围255以内
            if (Integer.parseInt(str) > 255)
                continue;
            list.add(str);
            //left到right后面一位
            dfs(s, step + 1, right + 1, res, list);
            list.removeLast();
        }
    }
}

```

### 279 二叉树的所有路径

**思路**：此题只需要使用深度优先遍历算法，每次遍历到一个节点就把该节点的值存入string中，然后判断是否为叶子节点，如果为叶子节点就需要把该条路路径添加进path中。本题需要注意string的用法，并且要注意对于string而言，我们采用的赋值传值的方式（不是引用或者指针），这样就意味着每次调用这个函数就会创建一个新的string，并且还会在调用的时候把上一个string的值传给它，这样就意味着每个函数中的path均不相同

- string带有隐藏回溯

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    public List<String> binaryTreePaths(TreeNode root) {
        LinkedList<String> res = new LinkedList<>();
        String path = "";
        if (root == null)
            return res;
        dfs(root, res,path);
       
        return res;
    }

    private void dfs(TreeNode root, LinkedList<String> res,String path) {
        if (root == null) {
            return;
        }
        if (root.left == null && root.right == null) {
            res.add(path + root.val);
        }
        dfs(root.left, res,path + root.val + "->");
        dfs(root.right, res, path + root.val + "->");
    }
}

```

- 用链表，回溯明显

```java
class Solution {
    public static List<String> binaryTreePaths(TreeNode root) {
        LinkedList<String> res = new LinkedList<>();
        if (root == null)
            return res;
        LinkedList<String> path = new LinkedList<>();
        // String path = "";
        dfs(root, res, path);
        return res;
    }

    private static void dfs(TreeNode root, LinkedList<String> res,LinkedList<String> path) {
        if (root == null) {
            return;
        }
        path.add(root.val + "");
        
        if (root.left == null && root.right == null) {
            res.add(String.join("->", path));
        }
        dfs(root.left, res,path);
        dfs(root.right, res, path);
        path.removeLast();
    }
}

```

## 动态规划

`labuladong`公众号

**DP table 是自底向上求解，递归解法是自顶向下求解** 

- 第一步：明确两点，“状态”和“选择”
- 第二步：明确`dp`数组定义
- 第三步：根据“选择”，思考状态转移的逻辑

```
for 状态1 in 状态1的所有取值:
	for 状态2 in 状态2的所有取值:
		for ...
			dp[状态1][状态2][...] = 择优(选择1,选择2,...)

```



#### 10 正则表达式匹配

```
举例: s = "aab", p = ".*"

```

```java
class Solution {
    /**
     * dp[i][j]表示 s[0:i-1]和p[0:j-1]的匹配情况 
     * 情况1：普通匹配：p[j] != '*' : dp[i][j] = dp[i-1][j-1] && (s[i] == p[j] || p[j] == '.') 
     * 情况2: p[j] == '*'，看前面一个字符的匹配情况
     * 2.1: '*'匹配 0 个，dp[i][j] = dp[i][j-2] 
     * 2.2: '*'匹配 1 个，dp[i][j] = dp[i-1][j-2] && (s[i] == p[j-1] || p[j-1] == '.') 
     * 2.3 '*'匹配 2 个，dp[i][j] = dp[i-2][j-2] && ((s[i] == p[j-1] && s[i-1] == p[j-1]) || p[j-1] == '.') 
     * ... '*'匹配 n 个，dp[i][j] = dp[i-n][j-2] && ((s[i-(n-1):i] 匹配 p[j-2]) || p[j-1] == '.') 
     * 2.1 2.2 2.3...合并可得:dp[i][j] = dp[i][j-2] || (dp[i-1][j-2] && s[i] 匹配 p[j-1]) && (dp[i-2][j-2] && s[i-1:i] 匹配 p[j-1])... 
     * 将 i = i - 1代入可得 dp[i-1][j] = dp[i-1][j-2] || (dp[i-2][j-2] && s[i-1] 匹配 p[j-1]) && (dp[i-3][j-2] && s[i-2:i] 匹配 p[j-1])... 
     * dp[i][j] 与 dp[i-1][j] 整体相差了 (s[i] 匹配 p[j-1])，以后每个 item 都相差 s[i] 匹配 p[j-1] 
     * 则 dp[i][j] = dp[i][j-2] || (dp[i-1][j] && (s[i] == p[j-1] || p[j-1] == '.'))
     */
    public boolean isMatch(String s, String p) {
        // 技巧：往原字符头部插入空格，这样得到 char 数组是从 1 开始，而且可以使得 dp[0][0] = true，可以将 true 这个结果滚动下去
        int m = s.length(), n = p.length();
        s = " " + s;
        p = " " + p;
        char[] ss = s.toCharArray();
        char[] pp = p.toCharArray();
        boolean[][] dp = new boolean[m + 1][n + 1];
        dp[0][0] = true;
        for (int i = 0; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // 如果下一个字符是 '*',则代表当前字符不能被单独使用，跳过
                if (j + 1 <= n && pp[j + 1] == '*')
                    continue;
                if (i >= 1 && pp[j] != '*') {
                    dp[i][j] = dp[i - 1][j - 1] && (ss[i] == pp[j] || pp[j] == '.');
                } else if (pp[j] == '*') {
                    dp[i][j] = (j >= 2 && dp[i][j - 2]) || (i >= 1 && dp[i - 1][j] && (ss[i] == pp[j - 1] || pp[j - 1] == '.'));
                }
            }
        }
        return dp[m][n];
    }
}

```

**python版本**

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        m, n = len(s), len(p)
        s = " " + s
        p = " " + p
        dp = [[False] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if j + 1 <= n and p[j + 1] == '*': 
                    continue
                if i - 1 >= 0 and p[j] != '*':
                    dp[i][j] = dp[i - 1][j - 1] and (s[i] == p[j] or p[j] == '.')
                elif p[j] == '*':
                    dp[i][j] = (j - 2 >= 0 and dp[i][j - 2]) or (i - 1 >= 0 and dp[i-1][j] and (s[i] == p[j-1] or p[j-1] == '.'))
        return dp[m][n]

```

**递归法**

递归算法的时间复杂度就是子问题个数乘以函数本身的复杂度

```python
class Solution:
    # 被 lru_cache 修饰的函数在被相同参数调用的时候，后续的调用都是直接从缓存读结果，而不用真正执行函数
    @lru_cache(None)
    # @functools.lru_cache()
    def isMatch(self, s: str, p: str) -> bool:
        # 结束条件, not p 表示 p 为空字符串，意味着递归结束
        if not p:
            return not s
        first_match = (len(s) > 0) and (p[0] in {s[0], '.'})
        if len(p) >= 2 and p[1] == '*':
            # '*'匹配0个或多个字符，右移 s 来达到匹配多个字符的效果
            return self.isMatch(s, p[2:]) or (first_match and self.isMatch(s[1:], p))
        else:
            return first_match and self.isMatch(s[1:], p[1:])

```

**加备忘录**

```python
class Solution:
    # @functools.lru_cache()
    def isMatch(self, s: str, p: str) -> bool:
        if not p:
            return not s
        self.memo = {}
        self.dp(s, p)
        return self.memo[(s, p)]

    def dp(self, s, p):
        # 结束条件, not p 表示 p 为空字符串，意味着递归结束
        if not p:
            return not s
        if (s, p) in self.memo:
            return self.memo[(s, p)]
        first_match = (len(s) > 0) and (p[0] in {s[0], '.'})
        if len(p) >= 2 and p[1] == '*':
            ans = self.dp(s, p[2:]) or (first_match and self.dp(s[1:], p))
        else:
            ans = first_match and self.dp(s[1:], p[1:])
        self.memo[(s, p)] = ans
        return ans

```

**注意嵌套函数时不要在参数里加`self`**

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        memo = dict()

        def dp(i, j):
            if (i, j) in memo:
                return memo[(i, j)]
            if j == len(p): 
                return i == len(s)
                
            first = i < len(s) and p[j] in {s[i], '.'}

            if j <= len(p) - 2 and p[j + 1] == '*':
                ans = dp(i, j + 2) or first and dp(i + 1, j)
            else:
                ans = first and dp(i + 1, j + 1)
            memo[(i, j)] = ans
            return ans
        return dp(0, 0)

```

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        memo = {}
        
        def dp(s, p):
            if not p:
                return not s
            if (s, p) in memo:
                return memo[(s, p)]
            first = len(s) > 0 and p[0] in {s[0], '.'}
            if len(p) >= 2 and p[1] == '*':
                ans = dp(s, p[2:]) or first and dp(s[1:], p)
            else:
                ans = first and dp(s[1:], p[1:])
            memo[(s, p)] = ans
            return ans
            
        return dp(s, p)

```



#### 322 零钱兑换

```
凑成总金额所需的最少的硬币数
例：coins = [1, 2, 5], amount = 11

```

自底向上

```java
class Solution {
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);//初始化为amount + 1相当于初始化为无穷大
        dp[0] = 0;
        for (int i = 1; i < dp.length; i++) {
            for (int coin : coins) {
                if (i - coin < 0)
                    continue;
                dp[i] = Math.min(dp[i], 1 + dp[i - coin]);
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
}

```

### 动态规划与回溯

#### 494 目标和

```
给定一个非负整数数组，a1, a2, ..., an, 和一个目标数，S。现在你有两个符号 + 和 -。对于数组中的任意一个整数，你都可以从 + 或 -中选择一个符号添加在前面。
返回可以使最终数组和为目标数 S 的所有添加符号的方法数。
 
示例：

输入：nums: [1, 1, 1, 1, 1], S: 3
输出：5
解释：

-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3

一共有5种方法让最终目标和为3。

```

- 动态规划

- **思路**：把`nums`分成两个子集`A`和`B`，分别带包分配`+`的数和分配`-`的数，那么它们和`target`存在如下关系：

    ```
    sum(A) - sum(B) = target 
    sum(A) + sum(A) = target + sum(B) + sum(A) = target + sum(nums)
    sum(A) = (target + sum(nums)) / 2
    
    ```

    转为子集背包问题：`nums`中存在几个子集`A`，使得`A`中元素的和为`(target + sum(nums)) / 2` ?

```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum < target || ((sum + target) & 1) == 1)
            return 0;
        int n = nums.length;
        sum = (sum + target) / 2;
        int[][] dp = new int[n + 1][sum + 1];
        //背包容量为0，不装入就是唯一的装法
        for (int i = 0; i < n + 1; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i < n + 1; i++) {
            for (int j = 0; j < sum + 1; j++) {
                if (j < nums[i - 1])
                    dp[i][j] = dp[i - 1][j];
                else
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - nums[i - 1]];
            }
        }
        return dp[n][sum];
    }
}

```

- 状态压缩

```java
class Solution {
    public int findTargetSumWays(int[] nums, int target) {
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if (sum < target || ((sum + target) & 1) == 1)
            return 0;
        int n = nums.length;
        sum = (sum + target) / 2;
        int[] dp = new int[sum + 1];
        //背包容量为0，不装入就是唯一的装法
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            //倒序，避免之前的结果影响后面的结果，因为正序的话dp[j-nums[i]]已经更新过了，不再是上一行的dp值
            //我们要做到：在计算新的dp[j]的时候，dp[j]和dp[j-nums[i]]还是上一层for循环的结果
            //如[2,2,3,5],当i=1时,j=6时，dp[6]=dp[6]||dp[6-2],而dp[4]=dp[4]||dp[4-2]=true,所以dp[6]=true,但这显然是错的，前2个数字凑不出和为6
            for (int j = sum; j >= nums[i]; j--) {
                    dp[j] = dp[j] + dp[j - nums[i]];
            }
        }
        return dp[sum];
    }
}

```

- 回溯

时间复杂度`O(2^n)，n=len(nums)`

```java
class Solution {
    // Map<String, Integer> memo = new HashMap<>();
    public int findTargetSumWays(int[] nums, int target) {
        backtrack(nums, 0, target);
        return ans;
    }

    private int ans = 0;
    private void backtrack(int[] nums, int i, int rest) {
        if (i == nums.length) {
            if (rest == 0) {
                ans++;
            }
            return;
        }
        
        //给nums[i]选择-号
        rest += nums[i];
        backtrack(nums, i + 1, rest);
        //撤销选择
        rest -= nums[i];
        //nums[i]选择+号
        rest -= nums[i];
        backtrack(nums, i + 1, rest);
        rest += nums[i];
    }
}

```

- 记忆化搜索


时间复杂度`O(n*target)`，即状态`(i,rest)`的数量

```java
class Solution {
    Map<String, Integer> memo = new HashMap<>();
    public int findTargetSumWays(int[] nums, int target) {
        return backtrack(nums, 0, target);
    }

    private int ans = 0;
    private int backtrack(int[] nums, int i, int rest) {
        if (i == nums.length) {
            if (rest == 0) {
                return 1;
            }
            return 0;
        }
        //小技巧：转成字符串才能作为哈希表的键
        String key = i + "," + rest;
        if (memo.containsKey(key)) {
            return memo.get(key);
        }
        ans = backtrack(nums, i + 1, rest - nums[i]) + backtrack(nums, i + 1, rest + nums[i]);
        memo.put(key, ans);
        return ans;
    }
    
}

```



### 子序列类型

#### 5 最长回文子串

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # 字符串倒序
        if s[::-1] == s:
            return s
        n = len(s)
        left, maxLen = 0, 0
        dp = [[False] * n for _ in range(n)]
        
        for i in range(n):
            dp[i][i] = True
        
        for i in range(n - 2, -1, -1):
            for j in range(i+1, n):
                
                if j - i == 1:
                    dp[i][j] = s[i] == s[j]
                elif j - i > 1:
                    dp[i][j] = s[i] == s[j] and dp[i + 1][j - 1]
                if j - i + 1 > maxLen and dp[i][j]:
                    maxLen = j - i + 1
                    left = i
        # 有可能只有单个字符是回文字符串，如"ac"
        if maxLen == 0:
            return s[0]
        return s[left:left + maxLen]

```

**时间复杂度**:O(N^2)

**还有中心扩散法**，可以参考CSDN博客字符串专栏

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # 倒叙一样，直接返回
        if s[::-1] == s:
            return s
        n = len(s)
        if n < 2:
            return s
        maxLen =  0
        res = odd = even = maxDis = [0]*2
        def centerSpread(s: str, left: int, right: int) -> []:
            n = len(s)
            while left >= 0 and right < n and s[left] == s[right]:
                left -= 1
                right += 1
            # 返回 左边界，右边界
            return [left + 1, right - 1]
        for i in range(n - 1):
            # 中心点分单个字符和2个字符
            # 奇数长度
            odd = centerSpread(s, i, i)
            # 偶数长度
            even = centerSpread(s, i, i + 1)
            # if odd[1] < even[1]:
              #  maxDis = even
            # else:
              #  maxDis = odd
            maxDis = even if odd[1] < even[1] else odd
            if maxDis[1] - maxDis[0] + 1 > maxLen:
                maxLen = maxDis[1] - maxDis[0] + 1
                res = maxDis
        return s[res[0] : res[1]+1]

```



#### 79 编辑距离

**自底向上**

`dp[i][j]`表示`word1` 前 `i`个字符`word2` 前 `j`个字符相等所需的最少操作次数

删除操作: `dp[i - 1][j] + 1`

插入操作: `dp[i][j - 1] + 1`

替换操作: `dp[i - 1][j - 1] + 1`

自己画`dp`表

```java
class Solution {
    public int minDistance(String word1, String word2) {
        int len1 = word1.length();
        int len2 = word2.length();
        //dp[0][0] = 0
        int[][] dp = new int[len1 + 1][len2 + 1];
        //索引从 1 开始
        //word2为空字符串, word1 前 i 个字符需要操作 i 次
        for (int i = 1; i <= len1; i++) {
            dp[i][0] = i;
        }
        //word1为空字符串
        for (int j = 1; j <= len2; j++) {
            dp[0][j] = j;
        }
        char[] s1 = word1.toCharArray();
        char[] s2 = word2.toCharArray();
        for (int i = 1; i <= len1; i++) {
            for (int j = 1; j <= len2; j++) {
                //字符相等，跳过
                if (s1[i - 1] == s2[j - 1])
                    dp[i][j] = dp[i - 1][j - 1];
                else {
                    dp[i][j] =  Math.min(Math.min(dp[i][j - 1] + 1, dp[i - 1][j] + 1), dp[i - 1][j - 1] + 1);
                }
            }
        }
        return dp[len1][len2];
    }
}

```

**python版本**

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1],
                                   dp[i - 1][j - 1]) + 1
                
        return dp[m][n]


```



**自顶向下**: 记忆化递归

`dp(i, j)`代表 `word1`索引`0~i`，`word2`索引`0~j`的最小编辑距离 

```java
class Solution {
    private String word1 = "";
    private String word2 = "";

    public int minDistance(String word1, String word2) {
        this.word1 = word1;
        this.word2 = word2;
        Map<Pair, Integer> cache = new HashMap<>();
        return dp(word1.length() - 1, word2.length() - 1, cache);
    }

    private int dp(int i, int j, Map<Pair, Integer> cache){
        Pair p = new Pair(i, j);
        if (cache.containsKey(p))
            return cache.get(p);
        //base case, 走到头，另一个字符串需要 当前索引加1次操作才能相等
        //如：word1 走完，那么 word1 需要插入 j + 1 次，或者 word2 删除 j + 1 次
        if (i == -1)
            return j + 1;
        if (j == -1)
            return i + 1;
        if (word1.charAt(i) == word2.charAt(j))
            cache.put(p, dp(i - 1, j - 1, cache));
        else
            cache.put(p, min(dp(i - 1, j - 1, cache), dp(i - 1, j, cache), dp(i, j - 1, cache)) + 1);
        return cache.get(p);
    }

    private int min(int a, int b, int c) {
        return Math.min(Math.min(a, b), c);
    }
}

```

#### 300 最长递增子序列

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        dp = [1] * n
        # dp = []
        for i in range(n):
            # dp.append(1)
            for j in range(i):
                if(nums[j] < nums[i]):
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

```

**时间复杂度 O(N^2)**

- 贪心+二分法，参考自己的 CSDN 博客

`dp[i]`：长度为`i+1`时的最长递增子序列，这样可以用寻找左边界的二分法替换掉`top`数组中比`nums[i]`大的第一个元素，复杂度降为`O(NlogN)`

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        dp = [nums[0]]
        for num in nums:
            # 严格大于末尾元素的才能加到末尾
            if num > dp[-1]:
                dp.append(num)
            # 寻找大于等于 num 的下标最小的元素
            left, right = 0, len(dp)
            while left < right:
                mid = (left + right) // 2
                if dp[mid] >= num:
                    right = mid
                else:
                    left = mid + 1
            dp[right] = num
        return len(dp)

```



```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        top = [0] * len(nums)
        piles = 0
        for i in range(len(nums)):
            left, right = 0, piles
            while left < right:
                mid = (left + right) // 2 # 注意 / 的结果数据类型是 float
                if top[mid] >= nums[i]:
                    right = mid
                elif top[mid] < nums[i]:
                    left = mid + 1
            if left == piles:
                piles += 1
            top[left] = nums[i]
        return piles

```

#### 312 戳气球

```
有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。 这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

求所能获得硬币的最大数量。
 

示例 1：
输入：nums = [3,1,5,8]
输出：167
解释：
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167

示例 2：

输入：nums = [1,5]
输出：10

```

首先想到回溯算法，但复杂度太高，通不过（力扣上有笔记记录了这一方法）

因为想象解题空间`n*n`大小，第 `i` 行有`n - i`种选择，复杂度`n!`大小，复杂度排行`O(n)<O(nlogn)<O(n^2)<O(n^3)...<O(n!)<O(2^n)`

**动态规划**: `dp[i][j]=x`表示戳破`i`和`j`之间（开区间，不包括`i`和`j`）所有气球，可以获得的最高分数`x`

- 设`i`和`j`之间最后戳破的气球编号为`k`，`(i,j)`区间内`k`两边已经被戳破，左右两边互不干扰，绕开子问题不独立的困境

```java
class Solution {
    public int maxCoins(int[] nums) {
        int n = nums.length;
        //将nums数组移植到新数组中
        int[] coins = new int[n + 2];
        // coins[0]和coins[n+1]可以看作是两个虚拟气球
        coins[0] = coins[n + 1] = 1;
        for (int i = 1; i < n + 1; i++) {
            coins[i] = nums[i - 1];
        }
        // dp[i][j]:(i, j)之间得到的最多分数，注意开区间，因为最后要求的是 dp[0][n+1]，不包括边界
        // 通过base case和最终状态dp[0][n+1]确定遍历方向
        int[][] dp = new int[n + 2][n + 2];
        for (int i = n; i >= 0; i--) {
            for (int j = i + 1; j < n + 2; j++) {
                for (int k = i + 1; k < j; k++) {
                    // 将编号为 k 的气球最后戳破，因为 k 是最后戳破，所以(i, k)和(k, j)之间的已经戳破了
                    dp[i][j] = Math.max(dp[i][j], dp[i][k] + dp[k][j] + coins[i] * coins[k] * coins[j]);
                }
            }
        }
        return dp[0][n + 1];
    }
}

```



#### 354 信封嵌套

```
输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
输出：3
解释：最多信封的个数为 3, 组合为: [2,3] => [5,4] => [6,7]。

```

```java
[w, h] w 升序排列，h 降序排列，对 h 寻找最长子序列

```

```java
class Solution {
    public int maxEnvelopes(int[][] envelopes) {
        /* Arrays.sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) { // w相同的话 h 降序排列
                return o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0];
            }
        }); */
        Arrays.sort(envelopes, (a, b) -> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);
        int n = envelopes.length;
        int[] height = new int[n];
        for (int i = 0; i < n; i++) {
            height[i] = envelopes[i][1];
        }
        return LIS(height);
    }

    private int LIS(int[] nums) {
        int[] dp = {nums[0]};
        //扑克牌堆数，看labuladong
        int end = 0;
        for (int num : nums) {

            int left = 0, right = end;
            while (left < right) {
                int mid = (left + right) >>> 1;
                if (dp[mid] >= num)
                    right = mid;
                else
                    left = mid + 1;
            }
            //新开堆
            if (left == end)
                end++;
            dp[left] = num;
        }
        return end;
    }
}

```

```java
class Solution {
    public int maxEnvelopes(int[][] envelopes) {

        /* Arrays.sort(envelopes, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) { // w相同的话 h 降序排列
                return o1[0] == o2[0] ? o2[1] - o1[1] : o1[0] - o2[0];
            }
        }); */
        Arrays.sort(envelopes, (a, b) -> a[0] == b[0] ? b[1] - a[1] : a[0] - b[0]);
        int n = envelopes.length;
        int[] height = new int[n];
        for (int i = 0; i < n; i++) {
            height[i] = envelopes[i][1];
        }
        return LIS(height);
    }

    // 最长递增子序列长度
    // 二分搜索法

    private int LIS(int[] nums) {
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        int end = 0;
        for (int num : nums) {
            if (num > dp[end]) {
                dp[++end] = num;
            } else {

                int left = 0, right = end;
                while (left < right) {
                    int mid = (left + right) >>> 1;
                    if (dp[mid] >= num)
                        right = mid;
                    else
                        left = mid + 1;
                }
                dp[left] = num;
            }
        }
        return end + 1;
    }
    //第二种
    private int LIS(int[] nums) {
        int[] dp = {nums[0]};
        
        for (int num : nums) {
            if (num > dp[dp.length - 1]) {
                dp = Arrays.copyOf(dp, dp.length + 1);
                dp[dp.length - 1] = num;
            } else {

                int left = 0, right = dp.length;
                while (left < right) {
                    int mid = (left + right) >>> 1;
                    if (dp[mid] >= num)
                        right = mid;
                    else
                        left = mid + 1;
                }
                dp[left] = num;
            }
        }
        return dp.length;
    }
}

```

**python**

```python
from typing import List
import bisect
# @lc code=start
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        n = len(envelopes)
        # lambda表达式
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        dp = [envelopes[0][1]]
        for i in range(1, n):
            # 新用法
            pos = bisect.bisect_left(dp, envelopes[i][1])
            if pos == len(dp):
                dp.append(envelopes[i][1])
            else:
                dp[pos] = envelopes[i][1]
        return len(dp)

```

```python
from typing import List
import bisect_left
# @lc code=start
class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        if not envelopes:
            return 0
        # n = len(envelopes)
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        dp = []
        for w, h in envelopes:
            pos = bisect_left(dp, h)
            # 将扩容这种情况合并了
            # 例如：dp = [1,2,3,4,5],h=6, bisect_dp(dp,h) = 5
            # dp[5:5+1] = [6], dp扩容，添加了一个6
            dp[pos:pos+1] = [h]
        return len(dp)

```

#### 516 最长回文子序列长度

```java
class Solution {
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        //dp[i][j]表示 s[i:j]的最长回文子序列的长度
        int[][] dp = new int[n][n];
        //base case
        for (int i = 0; i < n; i++) {
            dp[i][i] = 1;
        }
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) {
                if (s.charAt(i) == s.charAt(j))
                    dp[i][j] = dp[i + 1][j - 1] + 2;
                else
                    dp[i][j] = Math.max(dp[i + 1][j], dp[i][j - 1]);
            }
        }
        return dp[0][n - 1];
    }
}

```

**状态压缩**

```java
class Solution {
    public int longestPalindromeSubseq(String s) {
        int n = s.length();
        //dp[i][j]表示 s[i:j]的最长回文子序列的长度
        int[] dp = new int[n];
        //base case
        for (int i = 0; i < n; i++) {
            dp[i] = 1;
        }
        for (int i = n - 2; i >= 0; i--) {
            // 更新一维数组时，dp[i+1][j-1]会被dp[i][j-1]覆盖
            int pre = 0;
            for (int j = i + 1; j < n; j++) {
                int temp = dp[j];
                if (s.charAt(i) == s.charAt(j))
                    dp[j] = pre + 2;
                else
                    dp[j] = Math.max(dp[j], dp[j - 1]);
                pre = temp;
            }
        }
        return dp[n - 1];
    }
}

```

**python**

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        dp = [1] * len(s)
        for i in range(len(s) - 1, -1, -1):
            pre = 0
            for j in range(i+1, len(s)):
                temp = dp[j]
                if s[i] == s[j]:
                    dp[j] = pre + 2
                else:
                    dp[j] = max(dp[j], dp[j-1])
                pre = temp
        return dp[len(s) - 1]

```

### 背包问题

#### 0-1 背包

给你一个可装载重量为`W`的背包和`N`个物品，每个物品有重量和价值两个属性。第`i`个物品的重量为`wt[i]`，价值为`val[i]`，最多能装的价值是多少?

```
举例：
N=3,W=4
wt=[2,1,3],val=[4,2,3]

```

```java
public class Main {
    public static void main(String[] args) {
        int N = 3, W = 4;
        int[] wt = new int[] { 2, 1, 3 };
        int[] val = new int[] { 4, 2, 3 };
        int maxVal = knapsack(W, N, wt, val);
        System.out.println(maxVal);

    }
	//第一步：明确两点，“状态”和“选择”:状态有两个，背包容量和可选择的物品；选择就是“装入背包”和“不装入背包”
    //第二步：明确dp数组定义
    //第三步：根据“选择”，思考状态转移的逻辑
    private static int knapsack(int W, int N, int[] wt, int[] val) {
        //dp[i][w]:对于前 i 个物品，当前背包容量为 w 时，可以装下的最大价值
        int[][] dp = new int[N + 1][W + 1];
        for (int i = 1; i < N + 1; i++) {
            for (int w = 1; w < W + 1; w++) {
                if (w < wt[i - 1])
                    //背包容量不够,只能选择不装入背包
                    dp[i][w] = dp[i - 1][w];
                else
                    //装入或不装入
                    //第 i 个物品的重量为 wt[i-1],价值为 val[i-1]
                    dp[i][w] = Math.max(dp[i - 1][w], dp[i - 1][w - wt[i - 1]] + val[i - 1]);
            }
        }
        return dp[N][W];
    }

}

```

### 子集背包问题

#### 416 分割等和子集

```
给你一个 只包含正整数 的 非空 数组 nums 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

示例 1：

输入：nums = [1,5,11,5]
输出：true
解释：数组可以分割成 [1, 5, 5] 和 [11] 。

```

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        //奇数，不能平分成和相等的两个子集
        if ((sum & 1) == 1)
            return false;
        int target = sum / 2;
        //dp[i][j]:对于前i个数字，存在一个子集的和恰好是j
        boolean[][] dp = new boolean[n + 1][target + 1];
        //对于前i个数字，不选取任何数字
        for (int i = 0; i < n + 1; i++) {
            dp[i][0] = true;
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= target; j++) {
                if (j < nums[i - 1])
                    dp[i][j] = dp[i - 1][j];
                else
                    dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
            }
        }
        return dp[n][target];
    }
}

```



- 状态压缩

```java
class Solution {
    public boolean canPartition(int[] nums) {
        int n = nums.length;
        int sum = 0;
        for (int num : nums) {
            sum += num;
        }
        if ((sum & 1) == 1)
            return false;
        int target = sum / 2;
        //奇数
        boolean[] dp = new boolean[target + 1];
        dp[0] = true;
        // i-1化为i，范围[0, n-1]
        for (int i = 0; i < n; i++) {
            //倒序，避免之前的结果影响后面的结果，因为正序的话dp[j-nums[i]]已经更新过了，不再是上一行的dp值
            //我们要做到：在计算新的dp[j]的时候，dp[j]和dp[j-nums[i]]还是上一层for循环的结果
            //如[2,2,3,5],当i=1时,j=6时，dp[6]=dp[6]||dp[6-2],而dp[4]=dp[4]||dp[4-2]=true,所以dp[6]=true,但这显然是错的，前2个数字凑不出和为6
            for (int j = target; j >= nums[i]; j--) {
                dp[j] = dp[j] || dp[j - nums[i]];
            }
        }
        return dp[target];
    }
}

```



### 完全背包问题

#### 518 零钱兑换II

```
给你一个整数数组 coins 表示不同面额的硬币，另给一个整数 amount 表示总金额。
请你计算并返回可以凑成总金额的硬币组合数。如果任何硬币组合都无法凑出总金额，返回 0 。
假设每一种面额的硬币有无限个。 

示例 1：

输入：amount = 5, coins = [1, 2, 5]
输出：4
解释：有四种方式可以凑成总金额：
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1

```

```java
class Solution {
    public int change(int amount, int[] coins) {
        int n = coins.length;
        //dp[i][j]:若只使用前i个物品，当背包容量为j时，有dp[i][j]种凑法
        int[][] dp = new int[n + 1][amount + 1];
        //无为而治，不做选择
        for (int i = 0; i < n + 1; i++) {
            dp[i][0] = 1;
        }
        for (int i = 1; i < n + 1; i++) {
            for (int j = 1; j < amount + 1; j++) {
                if (j < coins[i - 1])
                    dp[i][j] = dp[i - 1][j];
                else {
                    //想要用面值为2的硬币凑出金额5，那么如果知道了凑出金额3的方法，再加上一枚面额为2的硬币，不就可以凑出5了嘛
                    dp[i][j] = dp[i - 1][j] + dp[i][j - coins[i - 1]];
                }
            }
        }
        return dp[n][amount];
    }
}

```

- 状态压缩

```java
class Solution {
    public int change(int amount, int[] coins) {
        int n = coins.length;
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        for (int i = 0; i < n; i++) {
            //从coins[i]开始就行
            for (int j = coins[i]; j < amount + 1; j++) {
                dp[j] = dp[j] + dp[j - coins[i]];
            }
        }
        return dp[amount];
    }
}

```



### 打家劫舍系列

#### 198 线性排列

```
街上有一排房屋，用 nums 表示，nums[i] 表示第 i 间房子中的金额，不能取相邻房子的钱，你需要尽可能多地取钱
如:nums=[2,1,7,9,3,1],算法返回 12， 2+9+1=12 or 2+7+3=12

```

```java
public class Main {
    public static void main(String[] args) {
        int[] nums = new int[] { 2, 1, 7, 9 };
        int res = rob(nums);
        System.out.println(res);

    }

    private static int rob(int[] nums) {
        int n = nums.length;
        if (n == 0) return 0;
        if (n == 1)
            return nums[0];
        //dp[i]:从第 i 间房子开始做选择
        int[] dp = new int[n + 2];
         for (int i = n - 1; i >= 0; i--) {
            //技巧：初始化dp[n-1]=nums[n-1],dp[n-2]=max(nums[n-1],nums[n-2])
            dp[i] = Math.max(dp[i + 1], nums[i] + dp[i + 2]);
        }
        return dp[0];
        
        //第2种定义
        //dp[i]: nums[0:i]取出的最多的钱
        int[] dp = new int[n];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < n; i++) {
            dp[i] = Math.max(dp[i - 1], nums[i] + dp[i - 2]);
        }
        return dp[n - 1];
    }
}

```

- 状态压缩

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        # dp[i]:从 nums[i] 开始取钱（做选择）
        # dp = [0] * (n+2)
        dp_2, dp_1, dp_0 = 0, 0, 0
        for i in range(n - 1, -1, -1):
            dp_0 = max(dp_1, dp_2 + nums[i])
            dp_2 = dp_1
            dp_1 = dp_0
        return dp_0

```



#### 213 环形排列

**注**:首尾不能同时取钱

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1: 
            return nums[0]

        def robRange(nums, start, end):
            dp_2 = dp_1 = dp_0 = 0
            for i in range(end, start - 1, -1):
                dp_0 = max(dp_1, dp_2 + nums[i])
                dp_2 = dp_1
                dp_1 = dp_0
            return dp_0
		
        return max(robRange(nums, 0, n - 2), robRange(nums, 1, n - 1))

```

#### 337 树形排列

- 记忆化搜索

```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode() {}
 *     TreeNode(int val) { this.val = val; }
 *     TreeNode(int val, TreeNode left, TreeNode right) {
 *         this.val = val;
 *         this.left = left;
 *         this.right = right;
 *     }
 * }
 */
class Solution {
    Map<TreeNode, Integer> memo = new HashMap<>();
    public int rob(TreeNode root) {
        if (root == null)
            return 0;
        //利用备忘录消除重叠子问题
        if (memo.containsKey(root))
            return memo.get(root);
        //当前节点选择偷
        int rob_1 = root.val + (root.left == null ? 0 : rob(root.left.left) + rob(root.left.right))
                + (root.right == null ? 0 : rob(root.right.left) + rob(root.right.right));
        //当前节点选择不偷
        int rob_0 = rob(root.left) + rob(root.right);
        int res = Math.max(rob_0, rob_1);
        memo.put(root, res);
        return res;
    }
}

```

```python
class Solution:
    memo = {}
    def rob(self, root: TreeNode) -> int:
        if not root:
            return 0
        # 注意 memo 前加 self
        if root in self.memo:
            return self.memo[root]
        money = root.val
        if root.left:
            money += self.rob(root.left.left) + self.rob(root.left.right) 
        if root.right:
            money += self.rob(root.right.left) + self.rob(root.right.right)
        
        ans = max(money, self.rob(root.left) + self.rob(root.right))
        self.memo[root] = ans
        return ans

```

- 动态规划
- 树状`dp`

```
每个节点可选择偷或者不偷两种状态，根据题目意思，相连节点不能一起偷

当前节点选择偷时，那么两个孩子节点就不能选择偷了
当前节点选择不偷时，两个孩子节点只需要拿最多的钱出来就行(两个孩子节点偷不偷没关系)
我们使用一个大小为 2 的数组来表示 int[] result = new int[2];
0 代表不偷，1 代表偷

任何一个节点能偷到的最大钱的状态可以定义为

当前节点选择不偷：当前节点能偷到的最大钱数 = 左右孩子偷或不偷得到的钱=Math.max{left[0],left[1]} + Math.max{right[0],right[1]}
当前节点选择偷：当前节点能偷到的最大钱数 = 左孩子选择自己不偷时能得到的钱 + 右孩子选择不偷时能得到的钱 + 当前节点的钱数=left[0] + right[0]

```

```java
class Solution {
    public int rob(TreeNode root) {
        int[] result = dfs(root);
        return Math.max(result[0], result[1]);
    }
    private int[] dfs(TreeNode root){
        if (root == null) {
            return new int[2];
        }
        int[] result = new int[2];
        int[] left = dfs(root.left);
        int[] right = dfs(root.right);
        result[0] = Math.max(left[0],left[1]) + Math.max(right[0],right[1]);
        result[1] = root.val + left[0] + right[0];
        return result;
    }
}

```

```python
class Solution:
    memo = {}
    def rob(self, root: TreeNode) -> int:
        def robTree(root):
            if not root:
                return [0, 0]
            left = robTree(root.left)
            right = robTree(root.right)
            # 偷
            rob_1 = root.val + left[0] + right[0]
            # 不偷
            rob_0 = max(left[0], left[1]) + max(right[0], right[1])
            return [rob_0, rob_1]
        res = robTree(root)
        return max(res[0], res[1])

```



## 十大排序（程序员小灰）

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618561136868.png" alt="1618561136868" style="zoom:70%;" />

### 冒泡排序

 ![Alt](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE3LmNuYmxvZ3MuY29tL2Jsb2cvODQ5NTg5LzIwMTcxMC84NDk1ODktMjAxNzEwMTUyMjMyMzg0NDktMjE0NjE2OTE5Ny5naWY) 

优化版本

```java
public int[] bubbleSort(int[] nums){
    int sortBorder = nums.length - 1;
    int lastExchangeIdx = 0,temp = 0;
    for (int i = 0; i < nums.length; i++) {
        boolean isSorted = true;
        for (int j = 0; j < sortBorder; j++) {
            if (nums[j] > nums[j+1]) {
                temp = nums[j];
                nums[j] = nums[j+1];
                nums[j + 1] = temp;
                //有元素交换，不是有序
                isSorted = false;
                //把无序数列的边界更新为最后一次交换元素的位置
                lastExchangeIdx = j;
            }
        }
        sortBorder = lastExchangeIdx;
        //已经有序，提前结束
        if (isSorted)
            break;
    }
    return nums;
}

```

#### 鸡尾酒排序（冒泡排序进化）

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618408027608.png" alt="1618408027608" style="zoom:50%;" />

```java
private static void sort(int[] nums) {
    // 轮数减半
    int border = nums.length - 1, lastExchangeIdx = 0;
    int reverseBorder = 0, lastExchangeIdxReverse = 0;
    for (int i = 0; i < nums.length / 2; i++) {
        int temp = 0;
        boolean isSorted = true;
        for (int j = 0; j < border; j++) {
            if (nums[j] > nums[j + 1]) {
                temp = nums[j + 1];
                nums[j + 1] = nums[j];
                nums[j] = temp;
                isSorted = false;
                lastExchangeIdx = j;
            }
        }
        border = lastExchangeIdx;
        if (isSorted)
            break;
        // 注意重新设为 true
        isSorted = true;
        for (int j = nums.length - i - 1; j > reverseBorder; j--) {
            if (nums[j] < nums[j - 1]) {
                temp = nums[j - 1];
                nums[j - 1] = nums[j];
                nums[j] = temp;
                isSorted = false;
                lastExchangeIdxReverse = j;
            }
        }
        reverseBorder = lastExchangeIdxReverse;
        if (isSorted)
            break;
    }
}

```



### 选择排序 ![Alt](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE3LmNuYmxvZ3MuY29tL2Jsb2cvODQ5NTg5LzIwMTcxMC84NDk1ODktMjAxNzEwMTUyMjQ3MTk1OTAtMTQzMzIxOTgyNC5naWY)

```java
int[] a ={56,89,594,41,7};
/****************选择排序 升序**************/
for ( int i = 0; i < a.length - 1; i++ ){
    for ( int j = i+1; j < a.length; j++ ){
        if ( a[j] < a[i] ){
            int temp = a[j];
            a[j] = a[i];
            a[i] = temp;
        }
    }
}
//也可以减少交换次数
private void selectionSort(int[] nums) {
    for (int i = 0; i < nums.length - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < nums.length; j++) {
            minIdx = nums[j] < nums[minIdx] ? j : minIdx;
        }
        int temp = nums[minIdx];
        nums[minIdx] = nums[i];
        nums[i] = temp;
    }
}

```

### 插入排序

 ![Alt](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE3LmNuYmxvZ3MuY29tL2Jsb2cvODQ5NTg5LzIwMTcxMC84NDk1ODktMjAxNzEwMTUyMjU2NDUyNzctMTE1MTEwMDAwMC5naWY) 

```java
private static int[] sort(int[] nums) {
    int temp = 0;
    for (int i = 1; i < nums.length; i++) {
        for (int j = i; j > 0 && nums[j] < nums[j - 1]; j--) {
            temp = nums[j-1];
            nums[j-1] = nums[j];
            nums[j] = temp;
        }
    }
    return nums;
}

```

**小灰优化**：减少交换，前一项复制到后一项，直接插入	

```java
private static int[] sort(int[] nums) {
    // int temp = 0;
    for (int i = 1; i < nums.length; i++) {
        int insertValue = nums[i];
        int j = i;
        for (; j > 0 && insertValue < nums[j - 1]; j--) {
            nums[j] = nums[j-1];
        }
        nums[j] = insertValue;
    }
    return nums;
}

```

### 希尔排序

 ![Alt](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE4LmNuYmxvZ3MuY29tL2Jsb2cvMTE5MjY5OS8yMDE4MDMvMTE5MjY5OS0yMDE4MDMxOTA5NDExNjA0MC0xNjM4NzY2MjcxLnBuZw?x-oss-process=image/format,png) 

```java
private static int[] sort(int[] nums) {
    for ( int gap = nums.length / 2; gap >= 1; gap /= 2) {
        for (int i = gap; i < nums.length; i++) {
            // int temp = 0;
            //把nums[i]摘出来，举例 gap = 1,nums = [3,2,1],nums[1] = nums[1-1],如果直接在for循环里用nums[i]<nums[j-gap],11行用nums[j] = nums[i]，那相当于nums[1]=nums[0]=3
            int minIdx = nums[i];
            int j = i;
            for (; j >= gap && minIdx < nums[j - gap]; j -= gap) {
                nums[j] = nums[j - gap];
            }
            nums[j] = minIdx;
        }
        // gap /= 2;
    }
    return nums;
}

```

### 快速排序

时间复杂度**O(nlogn)**，空间复杂度**O(logn)**，即递归栈所占空间

 ![Alt](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8yNTI3MzczLTJhMDYwZDBjOTk4MDU2NGQuZ2lm)

快速排序使用**分治法**来把一个串（list）分为两个子串（sub-lists）。具体算法描述如下：

- 从数列中挑出一个元素，称为 “基准”（pivot）；

- 重新排序数列，所有元素比基准值小的摆放在基准前面，所有元素比基准值大-的摆在基准的后面（相同的数可以到任一边）。在这个分区退出之后，该基准就处于数列的中间位置。这个称为分区（partition）操作；

- **递归**地（recursive）把小于基准值元素的子数列和大于基准值元素的子数列排序。 

#### 双边循环

从基准数对面开始

```java
private static void quiksort(int[] nums, int startIndex, int endIndex) {
    if (startIndex >= endIndex)
        return;
    int partition = partition(nums, startIndex, endIndex);
    quiksort(nums, startIndex, partition - 1);
    quiksort(nums, partition + 1, endIndex);
}

private static int partition(int[] nums, int start, int end) {
	//三数取中法
    /*int mid = start + (end - start) / 2;
    if (nums[start] > nums[mid])
        exchange(nums, start, mid);
    if (nums[mid] > nums[end])
        exchange(nums, mid, end);
    if (nums[start] > nums[end])
        exchange(nums, start, end);
    exchange(nums, start, mid);*/
    int pivot = nums[start];// 取第1个位置（也可以选择随机位置）的元素作为基准元素

    // int pivot = nums[start];
    int lo = start, hi = end;
    while (lo < hi) {
        //一定要先从右开始，举例，[4 1 3 5 6]，若从左开始，第一轮结束后左指针和右指针在5相遇，4和5交换，此时4的左边有5比4大，不符合左边都要比 pivot 小的要求
        while (lo < hi && nums[hi] >= pivot) {
            hi--;
        }
        while (lo < hi && nums[lo] <= pivot) {
            lo++;
        }

        if (lo < hi)
            exchange(nums, lo, hi);

    }
    exchange(nums, start, lo);
    return lo;
}

private static void exchange(int[] nums, int start, int end) {
    int temp = nums[start];
    nums[start] = nums[end];
    nums[end] = temp;
}

```

#### 单边循环

 

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618405126497.png" alt="1618405126497" style="zoom:50%;" />



<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618405258905.png" alt="1618405258905" style="zoom:80%;" />

```java
private static void quiksort(int[] nums, int startIndex, int endIndex) {
    if (startIndex >= endIndex)
        return;
    int partition = partition(nums, startIndex, endIndex);
    quiksort(nums, startIndex, partition - 1);
    quiksort(nums, partition + 1, endIndex);
}

private static int partition(int[] nums, int start, int end) {

    int pivot = nums[start];
    int mark = start;
    for (int i = start; i <= end; i++) {
        if (nums[i] < pivot) {
            mark++;
            exchange(nums, i, mark);
        }
    }
    exchange(nums, start, mark);
    return mark;
}

private static void exchange(int[] nums, int start, int end) {
    int temp = nums[start];
    nums[start] = nums[end];
    nums[end] = temp;
}

```

**用栈实现递归**

```java
private static void quicksort(int[] nums, int startIdx, int endIdx) {
    Stack<Map<String, Integer>> quickSortStack = new Stack<Map<String, Integer>>();
    Map<String, Integer> rootParam = new HashMap<>();
    rootParam.put("startIdx", startIdx);
    rootParam.put("endIdx", endIdx);
    quickSortStack.push(rootParam);
    while (!quickSortStack.isEmpty()) {
        Map<String, Integer> param = quickSortStack.pop();
        int pivotIdx = partition(nums, param.get("startIdx"), param.get("endIdx"));
        if (param.get("startIdx") < pivotIdx - 1) {
            Map<String, Integer> leftParam = new HashMap<>();
            leftParam.put("startIdx", param.get("startIdx"));
            leftParam.put("endIdx", pivotIdx - 1);
            quickSortStack.push(leftParam);
        }
        if (param.get("endIdx") > pivotIdx + 1) {
            Map<String, Integer> rightParam = new HashMap<>();
            rightParam.put("startIdx", pivotIdx + 1);
            rightParam.put("endIdx", param.get("endIdx"));
            quickSortStack.push(rightParam);
        }
    }
}

```



#### 填坑法

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618455810188.png" alt="1618455810188" style="zoom:80%;" />



<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618455947093.png" alt="1618455947093" style="zoom:80%;" />



<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618455991368.png" alt="1618455991368" style="zoom:80%;" />



<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618456080025.png" alt="1618456080025" style="zoom:80%;" />

```java
private static void quicksort(int[] nums, int startIdx, int endIdx) {
    if (startIdx >= endIdx)
        return;
    int pivot = partition(nums, startIdx, endIdx);
    quicksort(nums, startIdx, pivot - 1);
    quicksort(nums, pivot + 1, endIdx);
}

private static int partition(int[] nums, int start, int end) {
    int pivot = nums[start];
    int lo = start, hi = end, index = start;
    while (lo != hi) {
        while (lo < hi && nums[hi] >= pivot) {
            hi--;
        }
        if (lo < hi) {
            nums[lo] = nums[hi];
            lo++;
            index = hi;
        }
        while (lo < hi && nums[lo] <= pivot) {
            lo++;
        }
        if (lo < hi) {
            nums[hi] = nums[lo];
            hi--;
            index = lo;
        }
    }
    nums[index] = pivot;
    return index;
}

```

### 归并排序

```java
private static void mergesort(int[] nums, int startIdx, int endIdx) {
    if (startIdx >= endIdx)
        return;
    int mid = (startIdx + endIdx) / 2;
    mergesort(nums, startIdx, mid);
    mergesort(nums, mid + 1, endIdx);
    merge(nums, startIdx, mid, endIdx);
}

private static void merge(int[] nums, int lo, int mid, int hi) {
    // int mid = (lo + hi) / 2;
    int[] copy = new int[nums.length];
    //复制数组，从lo开始，长度 hi-lo+1
    System.arraycopy(nums, lo, copy, lo, hi - lo + 1);
    /*int i = lo, j = mid + 1;
    for (int k = lo; k <= hi; k++) {
        //必须先判断是否超出界限
        if (i > mid) {
            nums[k] = copy[j++];
        } else if (j > hi) {
            nums[k] = copy[i++];
        } else if (copy[i] > copy[j]) {
            nums[k] = copy[j++];
        } else if (copy[i] <= copy[j]) {
            nums[k] = copy[i++];
        }
    }*/
    int i = lo, j = mid + 1, k = lo;
    while (i <= mid && j <= hi) {
        if (copy[i] < copy[j]) {
            nums[k++] = copy[i++];
        } else {
            nums[k++] = copy[j++];
        }
    }
    //左侧有剩余
    while (i <= mid) {
        nums[k++] = copy[i++];
    }
    //右侧有剩余
    while (j <= hi) {
        nums[k++] = copy[j++];
    }
}

```

### 堆排序

最大堆：父结点大于子结点，挑较大的子结点交换

最小堆：父结点小于子结点，挑较小的子结点交换

升序：构建最大堆

降序：构建最小堆

**以最大堆为例，升序排列**

```java
public class Solution {
   
   public static void main(String[] args) {
      int[] nums = new int[] { 4, 6, 3, 2, 5, 5, 4 };
      heapsort(nums);
      System.out.println(Arrays.toString(nums));
   }

   private static void sink(int[] nums, int lo, int hi) {
      int temp = nums[lo];
      int child = 2 * lo + 1;
      int parent = lo;
      //注意是小于
      for (; child < hi; child = child * 2 + 1){
         if (child + 1 < hi && nums[child] < nums[child + 1])
            child++;
         if (temp < nums[child]) {
            nums[parent] = nums[child];
            parent = child;
         } else {
            break;
         }
      }
      nums[parent] = temp;
   }

   private static void heapsort(int[] nums) {
      //构建最大堆，从最后一个非叶子结点开始
      for (int i = (nums.length - 2) / 2; i >= 0; i--) {
         sink(nums, i, nums.length);
      }
      // System.out.println(Arrays.toString(nums));
      for (int i = nums.length - 1; i > 0; i--) {
         exchange(nums, 0, i);
         //因为条件判断里 child < hi,交换后的 i 结点不参与本轮操作
         sink(nums, 0, i);
      }
   }

   private static void exchange(int[] nums, int i, int j) {
      int temp = nums[i];
      nums[i] = nums[j];
      nums[j] = temp;
   }

}

```

堆排序算法步骤：

- 构建堆复杂度**O(n)**（需要数学证明，从最后一层开始，最多需要交换0,1,2...次，时间复杂度为 交换次数*每一层结点数）

- n - 1次循环，计算规模**(n-1)logn**，时间复杂度**O(nlogn)**

####  优先队列的实现

- **最大优先队列**，无论入队顺序如何，都是当前最大的元素优先出队
- 可以用**最大堆**来实现最大优先队列 ， 每一次入队操作就是堆的插入操作，每一次出队操作就是删除堆顶节点。 
- 上浮和下沉时间复杂度都为**O(logn)**

```java
public class PriorityQueue {
    private int[] nums;
    private int size;

    public PriorityQueue() {
        // this.size = size;
        nums = new int[1];
    }

    public static void main(String[] args) throws Exception{

        PriorityQueue priorityQueue = new PriorityQueue();
        priorityQueue.enQueue(3);
        priorityQueue.enQueue(5);
        priorityQueue.enQueue(10);
        priorityQueue.enQueue(2);
        priorityQueue.enQueue(7);
        System.out.println("出队元素：" + priorityQueue.deQueue());
        System.out.println("出队元素：" + priorityQueue.deQueue());
    }

    public void enQueue(int key) {
        if (size >= nums.length)
            resize();
        nums[size++] = key;//size初始为0
        swim();
    }

    public int deQueue() throws Exception {
        if (size <= 0) {
            throw new Exception("the queue is empty!");
        }
        int head = nums[0];
        nums[0] = nums[--size];
        sink();
        return head;
    }

    private void resize() {
        int newSize = this.size * 2;
        this.nums = Arrays.copyOf(this.nums, newSize);
    }

    public void swim() {
        int childIdx = size - 1;
        int parentIdx = (childIdx - 1) / 2;
        int temp = nums[childIdx];
        while (childIdx > 0 && temp > nums[parentIdx]) {
            nums[childIdx] = nums[parentIdx];
            childIdx = parentIdx;
            parentIdx = (parentIdx - 1）/ 2;
        }
        nums[childIdx] = temp;
    }

    public void sink() {
        int child = 1;
        int parentIdx = 0;
        int temp = nums[parentIdx];
        while(child < size) {
            if (child + 1 < size && nums[child] < nums[child + 1])
                child++;
            //不能把 temp < nums[child] 放 while 条件判断里，可以自己调试看看
            //因为 child 有可能右移，对于大顶堆来说 child 要选择更大的那个
            if(temp >= nums[child]) break;
            nums[parentIdx] = nums[child];
            parentIdx = child;
            child = 2 * child + 1;
        }
        nums[parentIdx] = temp;
    }
}

```

- **最小优先队列**， 无论入队顺序如何，都是当前最小的元素优先出 队  
- 可以用**最小堆**来实现最小优先队列，这样的话，每一次入队操作就是堆的插入操作，每一次出队操作就是删除堆顶节点 

- 上浮与下沉操作与最大堆互为**镜像**

```java
public void swim() {
    int childIdx = size - 1;
    int parentIdx = (childIdx - 1) / 2;
    int temp = nums[childIdx];
    while (childIdx > 0 && temp < nums[parentIdx]) {
        nums[childIdx] = nums[parentIdx];
        childIdx = parentIdx;
        parentIdx = parentIdx / 2;
    }
    nums[childIdx] = temp;
}

public void sink() {
        int child = 1;
        int parentIdx = 0;
        int temp = nums[parentIdx];
        while(child < size) {
            //小顶堆 child 选择更小的那个
            if (child + 1 < size && nums[child] > nums[child + 1])
                child++;
            if(temp <= nums[child]) break;
            nums[parentIdx] = nums[child];
            parentIdx = child;
            child = 2 * child + 1;
        }
        nums[parentIdx] = temp;
    }

```

### 计数排序

为了区分同数字的具体位置，先对**计数数组的值累计相加**，倒数遍历原数组，将数字放到计数数组的值**减一**的位置上，同时对应的计数数组的值减一，代表如果下一次遇到相同数字，位置减一

<img src="C:\Users\32332\AppData\Roaming\Typora\typora-user-images\1618575907813.png" alt="1618575907813" style="zoom:60%;" />

如索引9的值是5，代表数字9放到新数组的位置4上（索引需要减一）

```java
public class Solution {
    public static void main(String[] args) {
      int[] nums = new int[] { 4, 6, 5, 3, 5, 3 };
      Solution s = new Solution();
      int[] sortedArray = s.countSort(nums);
      System.out.println(Arrays.toString(sortedArray));
   }

   private int[] countSort(int[] nums) {
      int max = nums[0], min = nums[0];
      for (int i = 1; i < nums.length; i++) {
         max = Math.max(max, nums[i]);
         min = Math.min(min, nums[i]);
      }
       //计数数组
      int[] countArray = new int[max - min + 1];
      for (int j = 0; j < nums.length; j++) {
         countArray[nums[j] - min]++;
      }
       //统计应放到新数组的位置
      for (int i = 1; i < countArray.length; i++) {
         countArray[i] += countArray[i - 1];
      }
       //有序数组
      int[] sortedArray = new int[nums.length];
      for (int k = nums.length - 1; k >= 0; k--) {
         sortedArray[countArray[nums[k] - min] - 1] = nums[k];
         countArray[nums[k] - min]--;
      }
      return sortedArray;
   }
}

```

### 桶排序

```java
private int[] bucketSort(int[] nums,int bucketSize) {//bucketSize指桶的数字范围
    int max = nums[0], min = nums[0];
    for (int i = 1; i < nums.length; i++) {
        max = Math.max(max, nums[i]);
        min = Math.min(min, nums[i]);
    }
    //桶数量，定义为数组长度
    int bucketNum = nums.length;
    int bucketSize = (max - min) / (bucketNum - 1);
    ArrayList<LinkedList<Integer>> res = new ArrayList<LinkedList<Integer>>(bucketNum);
    for (int i = 0; i < bucketNum; i++) {
        res.add(new LinkedList<Integer>());
    }
    for (int i = 0; i < nums.length; i++) {
        //例如，bucketSize为10，数组为[10,20,...,90],80的索引为(80-10)/10 = 7
        int count = (nums[i] - min) / bucketSize;
        res.get(count).add(nums[i]);
    }
    for (int i = 0; i < bucketNum; i++) {
        //JDK底层使用归并排序或TimSort，时间复杂度O(nlogn)，这里每个桶元素分布均匀的话，排序总复杂度O(n)
        Collections.sort(res.get(i));
    }
    int index = 0;
    int[] sortedArray = new int[nums.length];
    for (List<Integer> list1 : res) {
        for (int i : list1) {
            sortedArray[index++] = i;
        }
    }
    return sortedArray;
}

```

### 基数排序

 ![Alt](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWFnZXMyMDE3LmNuYmxvZ3MuY29tL2Jsb2cvODQ5NTg5LzIwMTcxMC84NDk1ODktMjAxNzEwMTUyMzI0NTM2NjgtMTM5NzY2MjUyNy5naWY) 

 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523162128358.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pcmFjbGVvbg==,size_16,color_FFFFFF,t_70) 

```java
package DataStructureAndAlgorithms.Algorithms.Sort;

import java.util.Arrays;

/**
 * 基数排序
 */
public class RadixSort {
    public static void main(String[] args) {

        int[] a = {1, 5, 110, 55, 34, 85, 5, 34, 166, 24};
        radixSort(a);
        for (int value : a){
            System.out.print(value + " ");
        }
    }

    private static int[] expandArray(int[] arr, int value) {
        arr = Arrays.copyOf(arr, arr.length + 1);
        arr[arr.length - 1] = value;
        return arr;
    }

    private static int[] radixSort(int[] arr) {
        int maxValue = arr[0];
        for (int value : arr) {
            if (maxValue < value) {
                maxValue = value;
            }
        }
        //循环次数由位数决定
        for (int i = 1; maxValue / i > 0; i *= 10) {

            int[][] buckets = new int[10][0];//不可以是int[10][arr.length],与expandArray矛盾，会爆桶

            for (int j = 0; j < arr.length; j++) {
                int count = (arr[j] / i) % 10;//个、十、百、千位
                buckets[count] = expandArray(buckets[count],arr[j]);
            }

            int index = 0;
            for (int[] bucket : buckets) {
                for (int value : bucket) {
                    arr[index++] = value;
                }
            }

        }
        return arr;
    }
}

```

