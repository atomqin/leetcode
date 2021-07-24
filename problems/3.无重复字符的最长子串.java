import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-27 10:07:37
 * @LastEditTime: 2021-03-29 16:10:13
 */
/*
 * @lc app=leetcode.cn id=3 lang=java
 *
 * [3] 无重复字符的最长子串
 */

// @lc code=start
class Solution {
    public int lengthOfLongestSubstring(String s) {
        if (s.length() == 0 || s.length() == 1)
            return s.length();
        int[] window = new int[128];
        // Arrays.fill(window, -1);
        int left = 0, right = 0;
        int len = 0;
        while (right < s.length()) {
            char x = s.charAt(right);
            if(window[x] > 0)
                left = Math.max(left, window[x]);
            //右侧窗口下一个位置
            window[x] = right + 1;
            right++;
            len = Math.max(len, right - left);
        }
        return len;
    }
}
// @lc code=end
