/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-17 11:16:52
 * @LastEditTime: 2021-03-17 21:17:53
 */
/*
 * @lc app=leetcode.cn id=121 lang=java
 *
 * [121] 买卖股票的最佳时机
 */

// @lc code=start
class Solution {
    public static int maxProfit(int[] prices) {
        int n = prices.length;
        if (n < 2)
            return 0;
        
        int dp_i_0 = 0,dp_i_1 = -prices[0];
        for (int i = 1; i < n; i++) {
            dp_i_0 = Math.max(dp_i_0, dp_i_1 + prices[i]);
            dp_i_1 = Math.max(dp_i_1, -prices[i]);           
        }
        return dp_i_0;
    }
}
// @lc code=end
