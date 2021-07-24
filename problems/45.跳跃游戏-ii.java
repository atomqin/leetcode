/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-31 20:34:18
 * @LastEditTime: 2021-04-19 11:15:24
 */
/*
 * @lc app=leetcode.cn id=45 lang=java
 *
 * [45] 跳跃游戏 II
 */

// @lc code=start
class Solution {
    public int jump(int[] nums) {
        int jumps = 0;
        int farthest = 0;
        int end = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            farthest = Math.max(farthest, i + nums[i]);
            if (end == i) {
                jumps++;
                end = farthest;
            }
        }
        return jumps;
     }
}
// @lc code=end

