/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-10 19:24:09
 * @LastEditTime: 2021-04-11 09:56:52
 */
/*
 * @lc app=leetcode.cn id=34 lang=java
 *
 * [34] 在排序数组中查找元素的第一个和最后一个位置
 */

// @lc code=start
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
        int l = 0, r = nums.length - 1;
        while (l < r) {
            int m = (l + r) >>> 1;
            if (nums[m] < target) {
                l = m + 1;
            } else {
                r = m;
            }
        }
        if (nums[r] != target)
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
// @lc code=end
