/*
 * @lc app=leetcode.cn id=1217 lang=java
 *
 * [1217] 玩筹码
 */

// @lc code=start
class Solution {
    public int minCostToMoveChips(int[] position) {
        int i = 0;
        for (int p : position) {
            if ((p % 2) == 0)
                i++;
        }
        return Math.min(i, position.length - i);
    }
}
// @lc code=end

