/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-17 20:06:19
 * @LastEditTime: 2021-04-19 09:32:20
 */
/*
 * @lc app=leetcode.cn id=200 lang=java
 *
 * [200] 岛屿数量
 */

// @lc code=start
class Solution {
    
    public int numIslands(char[][] grid) {
        // int cnt = 0;
        int islandNum = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    dfs(grid, i, j);
                    islandNum++;
                }
            }
        }
        return islandNum;
    }

    private void dfs(char[][] grid,int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length ) {
            return;
        }
        if (grid[i][j] != '1')
            return;
        grid[i][j] = '2';
        dfs(grid, i - 1, j);
        dfs(grid, i + 1, j);
        dfs(grid, i, j - 1);
        dfs(grid, i, j + 1);
    }
}
// @lc code=end

