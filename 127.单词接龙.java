
/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-12 22:02:11
 * @LastEditTime: 2021-04-13 15:52:05
 */
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

/*
 * @lc app=leetcode.cn id=127 lang=java
 *
 * [127] 单词接龙
 */

// @lc code=start
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        Set<String> words = new HashSet<>(wordList);
        if (!words.contains(endWord))
            return 0;
        // words.add(beginWord);
        Queue<String> q1 = new LinkedList<>();
        Queue<String> q2 = new LinkedList<>();
        Set<String> visited1 = new HashSet<>();
        Set<String> visited2 = new HashSet<>();
        
        // -1 if not exsisted
        q1.add(beginWord);
        q2.add(endWord);
        visited1.add(beginWord);
        visited2.add(endWord);
        int cnt = 0;
        while (!q1.isEmpty()) {
            if (q1.size() > q2.size()) {
                Queue<String> temp = new LinkedList<>();
                temp = q1;
                q1 = q2;
                q2 = temp;
                Set<String> v = new HashSet<>();
                v = visited1;
                visited1 = visited2;
                visited2 = v;
            }
            int sz = q1.size();
            cnt++;
            while (sz-- > 0) {
                String cur = q1.poll();
                char[] chars = cur.toCharArray();
                // String cur = wordList.get(idx);
                for (int i = 0; i < cur.length(); i++) {
                    char c0 = chars[i];
                    for (char c = 'a'; c <= 'z'; c++) {
                        chars[i] = c;
                        String str = new String(chars);
                        // if (!canConvert(cur, s))
                        // continue;
                        if (!words.contains(str))
                            continue;
                        // int index = wordList.indexOf(str);
                        //访问过了
                        if (visited1.contains(str))
                            continue;
                        //相遇
                        if (visited2.contains(str))
                            return cnt + 1;
                        q1.add(str);
                        visited1.add(str);
                    }
                    chars[i] = c0;
                }
            }
        }
        return 0;
    }
}
// @lc code=end
