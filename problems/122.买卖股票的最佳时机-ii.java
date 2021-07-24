/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-17 21:56:21
 * @LastEditTime: 2021-04-01 20:35:36
 */
/*
 * @lc app=leetcode.cn id=122 lang=java
 *
 * [122] 买卖股票的最佳时机 II
 */

// @lc code=start
class Solution {
    public int maxProfit(int[] prices) {
        int len = prices.length;
        if (len < 2)
            return 0;
        int cash = 0, pre = 0, now = 0;
        for (int i = 1; i < prices.length; i++) {
            pre = prices[i - 1];
            now = prices[i];
            if (now - pre > 0)
                cash += now - pre;
        }
        return cash;
    }
}
// @lc code=end

