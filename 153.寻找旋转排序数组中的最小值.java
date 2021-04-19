/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-10 15:14:09
 * @LastEditTime: 2021-04-10 16:08:17
 */
import java.util.Arrays;

/*
 * @lc app=leetcode.cn id=153 lang=java
 *
 * [153] 寻找旋转排序数组中的最小值
 */

// @lc code=start
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
// @lc code=end

