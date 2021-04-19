/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-02 22:38:28
 * @LastEditTime: 2021-04-02 23:01:46
 */
/*
 * @lc app=leetcode.cn id=392 lang=java
 *
 * [392] 判断子序列
 */

// @lc code=start
class Solution {
    public boolean isSubsequence(String s, String t) {
        int index = -1;
        for (char c : s.toCharArray()) {
            //从 index + 1开始搜索下一次出现字符 c 的索引，没有返回 -1
            index = t.indexOf(c, index + 1);
            if (index == -1) {
                return false;
            }
        }
        return true;
    }
}
// @lc code=end

