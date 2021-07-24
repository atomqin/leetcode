/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-10 10:47:23
 * @LastEditTime: 2021-04-10 19:23:39
 */
/*
 * @lc app=leetcode.cn id=744 lang=java
 *
 * [744] 寻找比目标字母大的最小字母
 */

// @lc code=start
class Solution {
    public char nextGreatestLetter(char[] letters, char target) {
        /* if (target >= letters[letters.length - 1] || target < letters[0]) {
            return letters[0];
        } */
        int l = 0,r = letters.length - 1;
        while (l < r) {
            int mid = (l + r) >>> 1;
            if (letters[mid] > target) {
                r = mid;
            } else {
                l = mid + 1;
            }
        }
        return letters[l] > target ? letters[l] : letters[0];
    }
}
// @lc code=end

