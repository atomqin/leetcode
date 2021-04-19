import java.util.Map;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-28 15:20:16
 * @LastEditTime: 2021-03-28 19:39:03
 */
/*
 * @lc app=leetcode.cn id=76 lang=java
 *
 * [76] 最小覆盖子串
 */

// @lc code=start
class Solution {
    public String minWindow(String s, String t) {
        // 字母最大ASCII码小于128
        int[] need = new int[128];
        int[] window = new int[128];
        int left = 0, right = 0;
        int valid = 0;// 窗口中符合条件字符数
        int len = Integer.MAX_VALUE;
        String str = "";
        for (char c : t.toCharArray()) {
            need[c]++;
        }
        int n = 0;// t中不同字符数
        for (int i : need) {
            if (i == 0)
                continue;
            n++;
        }
        while (right < s.length()) {
            char x = s.charAt(right++);// 右侧窗口右移
            if (need[x] > 0) {
                window[x]++;
                if (window[x] == need[x])
                    valid++;
            }
            while (valid == n) {
                // 更新最小覆盖子串长度
                if (right - left < len) {
                    len = right - left;
                    str = s.substring(left, right);
                }
                char y = s.charAt(left++);// 左侧窗口右移
                if (need[y] > 0) {
                    window[y]--;
                    if (window[y] < need[y]) {
                        valid--;
                    }
                    // 和上面窗右侧右移对称
                    // if (window[y] == need[y]) {
                    // valid--;
                    // }
                    // window[y]--;
                }
            }
        }
        return str;
    }
}
// @lc code=end
