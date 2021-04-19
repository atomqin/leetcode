/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-02 19:37:45
 * @LastEditTime: 2021-04-02 22:04:26
 */
/*
 * @lc app=leetcode.cn id=605 lang=java
 *
 * [605] 种花问题
 */

// @lc code=start
class Solution {
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        int i = 0;
        while (i < flowerbed.length) {
            if (flowerbed[i] == 1) {
                i += 2;
            }
            // i 如果能到最后一个位置，说明前一个位置必定是 0，可以种植
            else if (i == flowerbed.length - 1 || flowerbed[i + 1] == 0) {
                i += 2;
                n--;
            } else {
                i += 3;
            }
            if (n <= 0)
                return true;
        }
        return false;
    }
}
// @lc code=end
