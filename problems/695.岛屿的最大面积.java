/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-13 16:06:49
 * @LastEditTime: 2021-04-13 16:40:56
 */
/*
 * @lc app=leetcode.cn id=695 lang=java
 *
 * [695] 岛屿的最大面积
 */

// @lc code=start
class Solution {

    // private int[][] dir = new int[][] { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 }
    // };

    public int maxAreaOfIsland(int[][] grid) {

        int maxRow = grid.length;
        int maxCol = grid[0].length;
        int maxArea = 0;
        for (int i = 0; i < maxRow; i++) {
            for (int j = 0; j < maxCol; j++) {
                maxArea = Math.max(maxArea, dfs(grid, i, j));
            }
        }
        return maxArea;
    }

    private int dfs(int[][] grid, int row, int col) {
        if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length || grid[row][col] == 0)
            return 0;
        grid[row][col] = 0;
        return 1 + dfs(grid, row + 1, col) + dfs(grid, row - 1, col) + dfs(grid, row, col + 1)
                + dfs(grid, row, col - 1);
    }
}
// @lc code=end
