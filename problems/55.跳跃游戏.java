/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-31 20:25:24
 * @LastEditTime: 2021-03-31 20:33:56
 */
/*
 * @lc app=leetcode.cn id=55 lang=java
 *
 * [55] 跳跃游戏
 */

// @lc code=start
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
// @lc code=end

