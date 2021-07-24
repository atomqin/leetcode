import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-12 15:48:55
 * @LastEditTime: 2021-04-12 21:56:57
 */
/*
 * @lc app=leetcode.cn id=279 lang=java
 *
 * [279] 完全平方数
 */

// @lc code=start
class Solution {
    public int numSquares(int n) {
        List<Integer> squares = generateSquares(n);
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        // int min = Integer.MAX_VALUE;
        for (int i = 1; i <= n; i++) {
            for (int square : squares) {
                if (i - square < 0)
                    break;
                dp[i] = Math.min(dp[i], dp[i - square] + 1);
            }

        }
        return dp[n];
    }
    //1 4 9...
    private List<Integer> generateSquares(int n) {
        List<Integer> squares = new ArrayList<>();
        int i = 1, increment = 3;
        while (i <= n) {
            squares.add(i);
            i += increment;
            increment += 2;
        }
        return squares;
    }
}
// @lc code=end
