给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有从根节点到叶子节点，路径总和等于给定目标和的路径。

- 写法一
```java
import java.util.*;
class Solution {
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        path.clear();
        res.clear();
        if(root == null) return res;
        path.add(root.val);
        traversal(root, targetSum);
        return res;
    }
    List<Integer> path = new LinkedList<>();
    List<List<Integer>> res = new ArrayList<>();
    private void traversal(TreeNode root, int sum){
        if (root == null) return;
        if (sum == root.val && root.left == null && root.right == null){
            //注意
            res.add(new ArrayList<>(path));
            return;
        }
        // if (root.left == null && root.right == null) return;
        if (root.left != null){
            path.add(root.left.val);
            traversal(root.left, sum - root.val);
            path.remove(path.size() - 1);
        }
        if (root.right != null){
            path.add(root.right.val);
            traversal(root.right, sum - root.val);
            path.remove(path.size() - 1);
        }
        
    }
}
```
- 写法二

```java
class Solution {
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<Integer> cur = new ArrayList<>();
        dfs(root, cur, 0, sum);            
        return res;
    }

    public void dfs(TreeNode node, List<Integer> cur, int sum, int target){
        if(node == null){
            return ;
        }
        cur.add(node.val);
        if(node.left == null && node.right == null && node.val + sum == target){;
            res.add(new ArrayList<>(cur));
            cur.remove(cur.size() - 1);
            return ;
        }            
        /如果没到达叶子节点，就继续从他的左右两个子节点往下找，注意到
        //下一步的时候，sum值要减去当前节点的值
        dfs(node.left, cur, sum + node.val, target);
        dfs(node.right, cur, sum + node.val, target);
        //我们要理解递归的本质，当递归往下传递的时候他最后还是会往回走，
        //我们把这个值使用完之后还要把它给移除，这就是回溯
        cur.remove(cur.size() - 1);
    }
}
```
