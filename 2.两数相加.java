/*
 * @Descripttion: 
 * @Author: atomqin
 * @Date: 2021-03-26 21:37:33
 * @LastEditTime: 2021-03-27 09:10:08
 */
/*
 * @lc app=leetcode.cn id=2 lang=java
 *
 * [2] 两数相加
 */

// @lc code=start
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode() {}
 *     ListNode(int val) { this.val = val; }
 *     ListNode(int val, ListNode next) { this.val = val; this.next = next; }
 * }
 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode curr = dummy;
        int sum = 0, carry = 0;
        // ListNode h1 = reverseListRecursion(l1);
        // ListNode h2 = reverseListRecursion(l2);
        while (l1 != null || l2 != null) {
            int x = (l1 == null) ? 0 : l1.val;
            int y = (l2 == null) ? 0 : l2.val;

            sum = x + y + carry;
            carry = sum / 10;
            curr.next = new ListNode(sum % 10);
            curr = curr.next;
            if (l1 != null)
                l1 = l1.next;
            if (l2 != null)
                l2 = l2.next;
        }
        if (carry > 0)
            curr.next = new ListNode(carry);
        return dummy.next;
    }

    /* private ListNode reverseListRecursion(ListNode head){
        if(head == null || head.next == null) return head;
        ListNode newHead = reverseListRecursion(head.next);
        head.next.next = head;
        head.next = null;
        return newHead;
    } */

}
// @lc code=end

