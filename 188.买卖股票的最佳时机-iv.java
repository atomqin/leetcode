import java.util.Arrays;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-02 10:24:28
 * @LastEditTime: 2021-04-02 19:27:47
 */
/*
 * @lc app=leetcode.cn id=188 lang=java
 *
 * [188] 买卖股票的最佳时机 IV
 */

// @lc code=start
class Solution {
    public int maxProfit(int k, int[] prices) {
        int[]buy = new int[k + 1];
        int[] sell = new int[k + 1];
        Arrays.fill(buy, Integer.MIN_VALUE);
        // Arrays.fill(sell, 0);
        for (int i = 0; i < prices.length; i++) {
            for (int j = 1; j <= k; j++) {
                buy[j] = Math.max(buy[j], sell[j - 1] - prices[i]);
                sell[j] = Math.max(sell[j], buy[j] + prices[i]);
            }
        }
        return sell[k];
    }
}
// @lc code=end

