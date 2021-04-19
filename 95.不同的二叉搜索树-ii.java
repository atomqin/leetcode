import java.util.List;

/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-04-06 08:55:18
 * @LastEditTime: 2021-04-06 15:18:40
 */
/*
 * @lc app=leetcode.cn id=95 lang=java
 *
 * [95] 不同的二叉搜索树 II
 */

// @lc code=start
/**
 * Definition for a binary tree node. public class TreeNode { int val; TreeNode
 * left; TreeNode right; TreeNode() {} TreeNode(int val) { this.val = val; }
 * TreeNode(int val, TreeNode left, TreeNode right) { this.val = val; this.left
 * = left; this.right = right; } }
 */
class Solution {
    public List<TreeNode> generateTrees(int n) {
        List<TreeNode>[] dp = new ArrayList[n + 1];
        dp[0] = new ArrayList<>();
        if (n == 0)
           return dp[0];
        dp[0].add(null);
        for (int len = 1; len <= n; len++) {
           dp[len] = new ArrayList<TreeNode>();
           for (int root = 1; root <= len; root++) {
              int left = root - 1;
              int right = len - root;
              for (TreeNode leftTree : dp[left]) {
                 for (TreeNode rightTree : dp[right]) {
                    TreeNode node = new TreeNode(root);
                    node.left = leftTree;
                    node.right = clone(rightTree, root);
                    dp[len].add(node);
                 }
              }
           }
        }
        return dp[n];
     }
  
     private TreeNode clone(TreeNode root, int offset) {
        if (root == null)
           return null;
        TreeNode node = new TreeNode(root.val + offset);
        node.left = clone(root.left, offset);
        node.right = clone(root.right, offset);
        return node;
     }
}

// @lc code=end
