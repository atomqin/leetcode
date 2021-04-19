import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-08 10:43:34
 * @LastEditTime: 2021-04-08 14:20:45
 */
/*
 * @lc app=leetcode.cn id=763 lang=java
 *
 * [763] 划分字母区间
 */

// @lc code=start
class Solution {
    public List<Integer> partitionLabels(String S) {
        int start = 0, end = 0;
        List<Integer> res = new ArrayList<>();
        int[] index = new int[26];
        for (int i = 0; i < S.length(); i++) {
            index[S.charAt(i) - 'a'] = i;
        }
        for (int i = 0; i < S.length(); i++) {
            end = Math.max(end, index[S.charAt(i) - 'a']);
            if (i == end) {
                res.add(end - start + 1);
                start = end + 1;
            }
        }
        return res;
    }
}
// @lc code=end
