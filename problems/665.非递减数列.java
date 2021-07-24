/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-08 09:58:08
 * @LastEditTime: 2021-04-08 10:28:37
 */
/*
 * @lc app=leetcode.cn id=665 lang=java
 *
 * [665] 非递减数列
 */

// @lc code=start
class Solution {
    public boolean checkPossibility(int[] nums) {
        if (nums.length < 2)
            return true;
        int cnt = 0;
        for (int i = 1; i < nums.length; i++) {
            if (nums[i] >= nums[i - 1])
                continue;
            cnt++;
            if (i >= 2 && nums[i] < nums[i - 2]) {
                nums[i] = nums[i - 1];
            } else {
                nums[i - 1] = nums[i];
            }
        }
        return cnt <= 1;
    }
}
// @lc code=end
