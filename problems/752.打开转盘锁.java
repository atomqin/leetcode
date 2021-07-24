import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-11 16:32:40
 * @LastEditTime: 2021-04-11 21:49:15
 */
/*
 * @lc app=leetcode.cn id=752 lang=java
 *
 * [752] 打开转盘锁
 */

// @lc code=start
class Solution {
    public int openLock(String[] deadends, String target) {
        // Set<String> deads = new HashSet<>();
        Set<String> deads = new HashSet<>();
        for (String string : deadends) {
            deads.add(string);
        }
        // ⽤集合不⽤队列，可以快速判断元素是否存在
        Set<String> visited = new HashSet<>();
        Set<String> q1 = new HashSet<>();
        Set<String> q2 = new HashSet<>();
        int step = 0;
        q1.add("0000");
        q2.add(target);
        while (!q1.isEmpty() && !q2.isEmpty()) {
            // 哈希集合在遍历的过程中不能修改，⽤ temp 存储扩散结果
            if (q1.size() > q2.size()) {
                Set<String> temp = new HashSet<>();
                temp = q1;
                q1 = q2;
                q2 = temp;
            }
            Set<String> temp = new HashSet<>();
            for (String s : q1) {
                if (deads.contains(s))
                    continue;
                if (q2.contains(s))
                    return step;
                visited.add(s);
                for (int j = 0; j < 4; j++) {
                    String up = plusOne(s, j);
                    if (!visited.contains(up))
                        temp.add(up);
                    String down = minusOne(s, j);
                    if (!visited.contains(down))
                        temp.add(down);
                }
            }
            step++;
            // temp 相当于 q1
            // 这⾥交换 q1 q2，下⼀轮 while 就是扩散 q2
            q1 = q2;
            q2 = temp;
        }

        return -1;
    }

    private String plusOne(String s, int j) {
        char[] ch = s.toCharArray();
        if (ch[j] == '9')
            ch[j] = '0';
        else
            ch[j] += 1;
        return new String(ch);
    }

    private String minusOne(String s, int j) {
        char[] ch = s.toCharArray();
        if (ch[j] == '0')
            ch[j] = '9';
        else
            ch[j] -= 1;
        return new String(ch);
    }
}
// @lc code=end
