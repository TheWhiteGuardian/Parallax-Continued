using System;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Burst.CompilerServices;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using System.Threading;

namespace Unity.Collections
{
    [NativeContainer]
    [NativeContainerIsAtomicWriteOnly]
    [NativeContainerSupportsDeallocateOnJobCompletion]
    public unsafe struct NativeAtomicInt : IDisposable
    {
        [BurstCompile]
        private struct DisposeJob : IJob
        {
            public NativeAtomicInt container;

            public void Execute() => container.Deallocate();
        }

#if ENABLE_UNITY_COLLECTIONS_CHECKS
        internal AtomicSafetyHandle m_Safety;

        [NativeSetClassTypeToNullOnSchedule]
        private DisposeSentinel m_DisposeSentinel;
#endif

        [NativeDisableUnsafePtrRestriction]
        private int* m_Data;

        private Allocator m_AllocatorLabel;

        public bool IsCreated
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => m_Data != null;
        }

        public NativeAtomicInt(Allocator allocator, int initialValue = 0)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            // Native allocation is only valid for Temp, Job and Persistent.
            if (allocator <= Allocator.None)
                throw new ArgumentException("Allocator must be Temp, TempJob or Persistent", nameof(allocator));

            DisposeSentinel.Create(out m_Safety, out m_DisposeSentinel, 1, allocator);
#endif
            m_Data = (int*)UnsafeUtility.Malloc(
                sizeof(int),
                UnsafeUtility.AlignOf<int>(),
                allocator
            );

            m_AllocatorLabel = allocator;

            if (Hint.Likely(IsCreated))
                *m_Data = initialValue;
        }

        public int Value
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

        /// <summary>
        /// Atomic abstraction of ++int.
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int IncrementPrefix() => IncrementPostfix() - 1;

        /// <summary>
        /// Atomic abstraction of int++.
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int IncrementPostfix()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            return Interlocked.Increment(ref *m_Data);
        }

        /// <summary>
        /// Atomic abstraction of --int.
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int DecrementPrefix() => DecrementPostfix() + 1;

        /// <summary>
        /// Atomic abstraction of int--.
        /// </summary>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int DecrementPostfix()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            return Interlocked.Decrement(ref *m_Data);
        }

        /// <summary>
        /// Atomically adds the given value to the internal integer value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer after this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Add(int value)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            return Interlocked.Add(ref *m_Data, value);
        }

        /// <summary>
        /// Atomically subtracts the given value from the internal
        /// integer value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer after this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Subtract(int value) => Add(-value);

        /// <summary>
        /// Atomically multiplies the internal integer by the given value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer after this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Multiply(int value)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = initialValue * value;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return computedValue;
        }

        /// <summary>
        /// Atomically divides the internal integer by the given value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer after this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Divide(int value)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = initialValue / value;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return computedValue;
        }

        /// <summary>
        /// Atomically negates the internal integer.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer after this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Negate()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = -initialValue;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return computedValue;
        }

        /// <summary>
        /// Atomically applies the binary NOT-gate to the internal integer.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer after this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Not()
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = ~initialValue;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return computedValue;
        }

        /// <summary>
        /// Atomically applies the binary AND-gate to the given value
        /// and the internal integer value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer BEFORE this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int And(int value)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = initialValue & value;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return initialValue;
        }

        /// <summary>
        /// Atomically applies the binary OR-gate to the given value
        /// and the internal integer value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer BEFORE this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Or(int value)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = initialValue | value;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return initialValue;
        }

        /// <summary>
        /// Atomically applies the binary XOR-gate to the given value
        /// and the internal integer value.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer BEFORE this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Xor(int value)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = initialValue ^ value;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return initialValue;
        }

        /// <summary>
        /// Atomically shifts the internal integer by 'value' bits to the left.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer BEFORE this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int ShiftLeft(int value)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = initialValue << value;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return initialValue;
        }

        /// <summary>
        /// Atomically shifts the internal integer by 'value' bits to the right.
        /// </summary>
        /// <param name="value"></param>
        /// <returns>The value of the internal integer BEFORE this operation
        /// completed.</returns>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int ShiftRight(int value)
        {
#if ENABLE_UNITY_COLLECTIONS_CHECKS
            AtomicSafetyHandle.CheckWriteAndThrow(m_Safety);
#else
            if (Hint.Unlikely(!IsCreated))
                return 0;
#endif

            int initialValue, computedValue;
            do
            {
                initialValue = *m_Data;
                computedValue = initialValue >> value;
            }
            while (initialValue != Interlocked.CompareExchange(ref *m_Data, computedValue, initialValue));
            return initialValue;
        }
    }
}
