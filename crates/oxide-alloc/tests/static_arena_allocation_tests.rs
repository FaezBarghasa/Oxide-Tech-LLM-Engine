use oxide_alloc::{DeviceMemoryArena, HostPinnedArena, TransactionalBlockTable};
use oxide_core::memory::DevicePtr;

#[test]
fn test_host_pinned_arena_allocation() {
    let mut arena = HostPinnedArena::new(1024 * 1024).expect("Failed to allocate HostPinnedArena");
    assert_eq!(arena.capacity(), 1024 * 1024);
    assert_eq!(arena.allocated_bytes(), 0);

    let slice1 = arena
        .alloc_slice::<u32>(1024)
        .expect("Slice 1 allocation failed");
    assert_eq!(slice1.len(), 1024);
    slice1[0] = 42;
    slice1[1023] = 999;
    assert_eq!(slice1[0], 42);
    assert_eq!(slice1[1023], 999);

    let allocated_after_1 = arena.allocated_bytes();
    assert!(allocated_after_1 >= 1024 * 4);

    arena.reset();
    assert_eq!(arena.allocated_bytes(), 0);
}

#[test]
fn test_device_memory_arena_slot_binding() {
    let mut fake_vram = vec![0u8; 1024 * 1024];
    let dev_ptr = unsafe { DevicePtr::from_raw(fake_vram.as_mut_ptr()) };

    let mut dev_arena = unsafe { DeviceMemoryArena::from_raw_device_ptr(dev_ptr, 1024 * 1024, 16) }
        .expect("DeviceMemoryArena creation failed");

    assert_eq!(dev_arena.num_slots(), 16);
    assert_eq!(dev_arena.slot_size_bytes(), 64 * 1024);

    let slot0_ptr = dev_arena.bind_slot(1001, 0).expect("Binding slot 0 failed");
    assert_eq!(slot0_ptr.as_device_address(), dev_ptr.as_device_address());

    // Double-binding same slot must fail
    assert!(dev_arena.bind_slot(1002, 0).is_err());

    // Out of bounds slot binding must fail
    assert!(dev_arena.bind_slot(1003, 16).is_err());

    // Release and rebind
    dev_arena.release_slot(0).expect("Release slot 0 failed");
    assert!(dev_arena.bind_slot(1004, 0).is_ok());
}

#[test]
fn test_transactional_speculative_rollback() {
    let mut table = TransactionalBlockTable::new(16);
    for block_id in 0..10 {
        table.push_block(block_id);
    }
    assert_eq!(table.physical_blocks().len(), 10);

    // Commit 64 accepted tokens (4 blocks of 16)
    table.commit_speculation(64);
    assert_eq!(table.active_tokens(), 64);

    // Rollback to 32 accepted tokens (2 blocks of 16 retained)
    table.rollback(32, 48);
    assert_eq!(table.active_tokens(), 32);
    assert_eq!(table.physical_blocks().len(), 2);
    assert_eq!(table.physical_blocks(), &[0, 1]);
}
