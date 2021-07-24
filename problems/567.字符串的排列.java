import java.util.Arrays;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-29 10:16:58
 * @LastEditTime: 2021-03-29 15:52:47
 */
/*
 * @lc app=leetcode.cn id=567 lang=java
 *
 * [567] 字符串的排列
 */

// @lc code=start
class Solution {
    public boolean checkInclusion(String s1, String s2) {
        if (s1.length() > s2.length())
            return false;
        int[] window = new int[128];
        int[] s1_map = new int[128];
        
        //窗口长度为 s1.length()
        for (int i = 0; i < s1.length(); i++) {
            s1_map[s1.charAt(i)]++;
            window[s2.charAt(i)]++;
        }
        if (Arrays.equals(window, s1_map))
            return true;
        int left = 0, right = s1.length();
        while (right < s2.length()) {
            window[s2.charAt(right)]++;
            right++;
            window[s2.charAt(left)]--;
            left++;
            if (Arrays.equals(window, s1_map))
                return true;
        }
        return false;
    }
}
// @lc code=end
