import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Set;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-11 09:27:07
 * @LastEditTime: 2021-04-12 20:25:30
 */
/*
 * @lc app=leetcode.cn id=1091 lang=java
 *
 * [1091] 二进制矩阵中的最短路径
 */

// @lc code=start
class Solution {
    public int shortestPathBinaryMatrix(int[][] grid) {
        int n = grid.length;
        if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1)
            return -1;
        if (n == 1)
            return 1;
        Queue<int[]> q = new LinkedList<>();
        q.add(new int[] { 0, 0 });
        int step = 1;
        int[][] dir = { { -1, -1 }, { -1, 0 }, { -1, 1 }, { 0, -1 }, { 0, 1 }, { 1, -1 }, { 1, 0 }, { 1, 1 } };
        while (!q.isEmpty()) {
            int sz = q.size();
            for (int i = 0; i < sz; i++) {
                int[] pos = q.poll();
                if (pos[0] == n - 1 && pos[1] == n - 1)
                    return step;

                for (int[] direction : dir) {
                    int pos_0 = pos[0] + direction[0];
                    int pos_1 = pos[1] + direction[1];

                    if (pos_0 < 0 || pos_1 < 0 || pos_0 > n - 1 || pos_1 > n - 1)
                        continue;

                    if (grid[pos_0][pos_1] == 1)
                        continue;
                        // 标记访问
                    grid[pos_0][pos_1] = 1;
                    
                    int[] newPos = new int[] { pos_0, pos_1 };
                    q.offer(newPos);
                }
            }
            step++;
        }
        return -1;
    }
}
// @lc code=end
