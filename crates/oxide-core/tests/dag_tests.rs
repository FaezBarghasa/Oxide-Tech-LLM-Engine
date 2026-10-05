use crossbeam_queue::SegQueue;
use oxide_core::dag::{SharedPhysicalBlock, TreeNode};

#[test]
fn test_dag_tree_node_hierarchy() {
    let root = TreeNode::new_root(1, 101, 0);
    assert_eq!(root.node_id, 1);
    assert!(root.parent.is_none());

    let child1 = root.create_child(2, 102, 0.95, 1);
    let child2 = root.create_child(3, 103, 0.91, 2);

    assert_eq!(child1.parent.as_ref().unwrap().node_id, 1);
    assert_eq!(child2.parent.as_ref().unwrap().node_id, 1);
}

#[test]
fn test_lock_free_cow_block_fork_release() {
    let free_queue = SegQueue::new();
    let block = SharedPhysicalBlock::new(42);

    // Initial refcount = 1
    assert_eq!(
        block.ref_count.load(std::sync::atomic::Ordering::Relaxed),
        1
    );

    // Fork twice for parallel branches
    block.fork();
    block.fork();
    assert_eq!(
        block.ref_count.load(std::sync::atomic::Ordering::Relaxed),
        3
    );

    // Release branch 1
    block.release(&free_queue);
    assert!(free_queue.is_empty());

    // Release branch 2
    block.release(&free_queue);
    assert!(free_queue.is_empty());

    // Release last reference -> returned to free_queue
    block.release(&free_queue);
    assert_eq!(free_queue.pop(), Some(42));
}
