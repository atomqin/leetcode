/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-10 11:08:21
 * @LastEditTime: 2021-04-10 14:45:24
 */
/*
 * @lc app=leetcode.cn id=540 lang=java
 *
 * [540] 有序数组中的单一元素
 */

// @lc code=start
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
// @lc code=end

