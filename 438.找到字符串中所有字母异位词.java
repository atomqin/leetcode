import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-29 20:04:51
 * @LastEditTime: 2021-03-29 21:47:01
 */
/*
 * @lc app=leetcode.cn id=438 lang=java
 *
 * [438] 找到字符串中所有字母异位词
 */

// @lc code=start
class Solution {
    public List<Integer> findAnagrams(String s, String p) {
        if (s.length() < p.length())
            return new ArrayList<>();
        List<Integer> res = new ArrayList<>();
        int[] window = new int[26];
        int[] p_map = new int[26];
        int left = 0, right = 0;
        
        for (int i = 0; i < p.length(); i++) {
            p_map[p.charAt(i) - 'a']++;
        }
        while (right < s.length()) {
            window[s.charAt(right) - 'a']++;
            while (window[s.charAt(right) - 'a'] > p_map[s.charAt(right) - 'a']) {
                window[s.charAt(left) - 'a']--;
                left++;
            }
            right++;
            if (right - left == p.length())
                res.add(left);
        }
        return res;
    }
}
// @lc code=end

