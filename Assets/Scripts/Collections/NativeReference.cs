using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Burst.CompilerServices;

namespace Unity.Collections
{
    namespace LowLevel.Unsafe
    {
        /// <summary>
        /// Same API as <see cref="NativeReference{T}"/>, but without checks
        /// against race conditions and memory leaks.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        public unsafe struct UnsafeReference<T> : IDisposable
            where T : unmanaged
        {
            [BurstCompile]
            private struct DisposeJob : IJob
            {
                public UnsafeReference<T> container;

                public void Execute() => container.Dispose();
            }

            [NativeDisableUnsafePtrRestriction]
            private T* m_Data;

            private Allocator m_AllocatorLabel;

            public bool IsCreated
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get => m_Data != null;
            }

            public UnsafeReference(Allocator allocator, NativeArrayOptions clearMemory = NativeArrayOptions.ClearMemory)
                : this(allocator)
            {
                if (Hint.Likely(IsCreated) && clearMemory == NativeArrayOptions.ClearMemory)
                {
                    UnsafeUtility.MemClear(m_Data, UnsafeUtility.SizeOf<T>());
                }
            }

            public UnsafeReference(T value, Allocator allocator)
                : this(allocator)
            {
                if (Hint.Likely(IsCreated))
                    *m_Data = value;
            }

            private UnsafeReference(Allocator allocator)
            {
                // Native allocation is only valid for Temp, Job and Persistent.
                if (allocator <= Allocator.None)
                {
                    // Invalid allocator.
                    m_AllocatorLabel = Allocator.Invalid;
                    m_Data = null;
                    return;
                }

                m_Data = (T*)UnsafeUtility.Malloc(
                    UnsafeUtility.SizeOf<T>(),
                    UnsafeUtility.AlignOf<T>(),
                    allocator
                );

                m_AllocatorLabel = allocator;
            }

            public T Value
            {
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                get
                {
                    if (Hint.Unlikely(!IsCreated)) return default;
                    return *m_Data;
                }
                [MethodImpl(MethodImplOptions.AggressiveInlining)]
                set
                {
                    if (Hint.Unlikely(!IsCreated)) return;
                    *m_Data = value;
                }
            }

            public void Dispose()
            {
                if (Hint.Likely(IsCreated))
                {
                    UnsafeUtility.Free(m_Data, m_AllocatorLabel);
                    m_Data = null;
                    m_AllocatorLabel = Allocator.None;
                }
            }

            public JobHandle Dispose(JobHandle inputDeps)
            {
                if (Hint.Likely(IsCreated))
                {
                    inputDeps = new DisposeJob()
                    {
                        container = this
                    }.Schedule(inputDeps);

                    // A scheduled job has a copy of the pointers.
                    m_Data = null;
                    m_AllocatorLabel = Allocator.None;
                }

                return inputDeps;
            }

            public ref T AsRef()
            {
                if (Hint.Unlikely(!IsCreated))
                    throw new NullReferenceException("Cannot get an as-ref value from a null pointer!");
                return ref *m_Data;
            }

            public ref readonly T AsRefReadOnly()
            {
                if (Hint.Unlikely(!IsCreated))
                    throw new NullReferenceException("Cannot get an as-ref value from a null pointer!");

                return ref *m_Data;
            }

            internal T* GetUncheckedPtr() => m_Data;
        }
    }

    [NativeContainer]
    [NativeContainerSupportsDeallocateOnJobCompletion]
    public unsafe struct NativeReference<T> : IDisposable
        where T : unmanaged
    {
        [BurstCompile]
        private struct DisposeJob : IJob
        {
            public NativeReference<T> container;

            public void Execute() => container.Deallocate();
        }

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        internal AtomicSafetyHandle m_Safety;

        [NativeSetClassTypeToNullOnSchedule]
        private DisposeSentinel m_DisposeSentinel;
#endif

        [NativeDisableUnsafePtrRestriction]
        private T* m_Data;

        private Allocator m_AllocatorLabel;

        public bool IsCreated
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => m_Data != null;
        }

        public NativeReference(Allocator allocator, NativeArrayOptions clearMemory = NativeArrayOptions.ClearMemory)
            : this(allocator, 2)
        {
            if (Hint.Likely(IsCreated) && clearMemory == NativeArrayOptions.ClearMemory)
            {
                UnsafeUtility.MemClear(m_Data, UnsafeUtility.SizeOf<T>());
            }
        }

        public NativeReference(T value, Allocator allocator)
            : this(allocator, 2)
        {
            if (Hint.Likely(IsCreated))
                *m_Data = value;
        }

        private NativeReference(Allocator allocator, int disposeSentinelStackDepth)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            // Native allocation is only valid for Temp, Job and Persistent.
            if (allocator <= Allocator.None)
                throw new ArgumentException("Allocator must be Temp, TempJob or Persistent", nameof(allocator));

            CollectionHelper.CheckIsUnmanaged<T>();

            DisposeSentinel.Create(out m_Safety, out m_DisposeSentinel, disposeSentinelStackDepth, allocator);
#endif
            m_Data = (T*)UnsafeUtility.Malloc(
                UnsafeUtility.SizeOf<T>(),
                UnsafeUtility.AlignOf<T>(),
                allocator
            );

            m_AllocatorLabel = allocator;
        }

        public T Value
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                // This does a null check and a race condition check.
                AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#else
                // We're not gonna deref a null pointer. No way.
                if (Hint.Unlikely(!IsCreated)) return default;
#endif
                return *m_Data;
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            set
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
                // Block write on null.
                if (Hint.Unlikely(!IsCreated)) return;
#endif
                *m_Data = value;
            }
        }

        private void Deallocate()
        {
            if (Hint.Likely(IsCreated))
            {
                UnsafeUtility.Free(m_Data, m_AllocatorLabel);
                m_Data = null;
                m_AllocatorLabel = Allocator.None;
            }
        }

        public void Dispose()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            DisposeSentinel.Dispose(ref m_Safety, ref m_DisposeSentinel);
#endif
            Deallocate();
        }

        public JobHandle Dispose(JobHandle inputDeps)
        {
            if (Hint.Likely(IsCreated))
            {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
                DisposeSentinel.Clear(ref m_DisposeSentinel);
                // Don't destroy AtomicSafetyHandle! Unity first needs it to
                // schedule the disposal job correctly.
#endif
                inputDeps = new DisposeJob()
                {
                    container = this
                }.Schedule(inputDeps);

#if ENABLE_UNITY_COLLECTIONS_CHECKS
                AtomicSafetyHandle.Release(m_Safety);
#endif
                // A scheduled job has a copy of the pointers.
                m_Data = null;
                m_AllocatorLabel = Allocator.None;
            }

            return inputDeps;
        }

        public ref T AsRef()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                throw new NullReferenceException("Cannot get an as-ref value from a null pointer!");
#endif
            return ref *m_Data;
        }

        public ref readonly T AsRefReadOnly()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                throw new NullReferenceException("Cannot get an as-ref value from a null pointer!");
#endif
            return ref *m_Data;
        }

        internal T* GetUnsafePtr()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#endif
            return m_Data;
        }

        internal T* GetUnsafeReadOnlyPtr()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckReadAndThrow(m_Safety);
#endif
            return m_Data;
        }

        internal T* GetUncheckedPtr() => m_Data;
    }
}
