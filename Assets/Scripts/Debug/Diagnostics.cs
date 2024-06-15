using UnityEngine;
using UnityEngine.UI;
using Unity.Collections;
using Unity.Jobs;
using Unity.Burst;
using Unity.Mathematics;
using Unity.Collections.LowLevel.Unsafe;
using System.Threading;

[RequireComponent(typeof(LineRenderer), typeof(Camera))]
[DefaultExecutionOrder(100)]
public sealed class Diagnostics : MonoBehaviour
{
    internal static void Setup()
    {
        if (instance == null)
        {
            int layer = 4;
            int mask = 1 << layer;

            GameObject obj = new GameObject("Diagnostics Renderer");
            obj.layer = layer;
            
            // Setup overlay camera.
            Camera cam = obj.AddComponent<Camera>();
            cam.orthographic = true;
            cam.transform.position = new Vector3(0f, -100f, 0f);
            cam.depth = 10;
            cam.clearFlags = CameraClearFlags.Nothing;
            cam.orthographicSize = 2f;
            cam.cullingMask = mask;
            cam.nearClipPlane = 0.001f;
            cam.farClipPlane = 10f;

            var line = obj.AddComponent<LineRenderer>();
            line.useWorldSpace = false;
            var mat = new Material(Shader.Find("Unlit/Color"));
            mat.SetColor("_Color", Color.red);
            line.sharedMaterial = mat;
            line.widthMultiplier = 0.01f;

            obj.AddComponent<Diagnostics>();
        }
    }

    internal static void PushTime(float time)
    {
        // Block writes if the array is in use.
        if (!positionBakeJob.IsCompleted)
            return;

        largestTime = Mathf.Max(time, largestTime);
        times[cursor] = time;
        cursor++;
        cursor = cursor < times.Length ? cursor : 0;
    }

    private static Diagnostics instance;
    private static NativeArray<float> times;
    private static float largestTime = 0f;
    private static int cursor = 0;
    private static JobHandle positionBakeJob;
    internal static float MeanOfSamples { get; private set; }
    internal static float MaxOfSamples { get; private set; }
    internal static float MinOfSamples { get; private set; }

    private static float widthScale = 0.2f;
    private static float heightScale = 0.2f;

    [SerializeField, Min(1)]
    private int timeSampleCount = 250;

    private Camera cam;
    private LineRenderer line;
    private NativeArray<Vector3> linePositions;
    private NativeReference<float> timesSum;
    private NativeReference<float> timesMin;
    private NativeReference<float> timesMax;

    private void Awake()
    {
        if (instance == null)
        {
            instance = this;
            cam = GetComponent<Camera>();
            times = new NativeArray<float>(timeSampleCount, Allocator.Persistent);
            line = GetComponent<LineRenderer>();
            line.positionCount = timeSampleCount;
            linePositions = new NativeArray<Vector3>(line.positionCount, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            timesSum = new NativeReference<float>(Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            timesMin = new NativeReference<float>(Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
            timesMax = new NativeReference<float>(Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        }
        else enabled = false;
    }

    private void OnDestroy()
    {
        if (instance != this) return;

        // Ensure dependent job has completed.
        positionBakeJob.Complete();
        times.Dispose();
        linePositions.Dispose();
        timesSum.Dispose();
    }

    // Guaranteed to run after other Update methods.
    private void Update()
    {
        float halfHeight = cam.orthographicSize;
        float halfWidth = halfHeight * Screen.width / Screen.height;

        timesSum.Value = 0f;
        timesMin.Value = 1000000f;
        timesMax.Value = -100f;

        positionBakeJob = new PositionBakeJob()
        {
            times = times,
            cursor = cursor,
            normalizers = new float2(1f / times.Length, 1f / Mathf.Max(0.001f, largestTime)),
            min = new float2(-1f * halfWidth, Mathf.Lerp(1f, -1f, heightScale) * halfHeight),
            max = new float2(Mathf.Lerp(-1f, 1f, widthScale) * halfWidth, 1f * halfHeight),
            linePositions = linePositions.Reinterpret<float3>(),
            sum = timesSum,
            maxT = timesMax,
            minT = timesMin
        }.ScheduleBatch(times.Length, 32);
    }

    private void LateUpdate()
    {
        positionBakeJob.Complete();
        for (int i = 0; i < times.Length; i++)
        {
            line.SetPosition(i, linePositions[i]);
        }
        MeanOfSamples = timesSum.Value / times.Length;
        MaxOfSamples = timesMax.Value;
        MinOfSamples = timesMin.Value;
        // Slowly bring the largest time back down.
        largestTime = Mathf.Lerp(largestTime, MaxOfSamples, Time.deltaTime * 0.5f);
    }

    [BurstCompile]
    private struct PositionBakeJob : IJobParallelForBatch
    {
        [ReadOnly]
        public NativeArray<float> times;

        public int cursor;
        // Normalizers for the data in the X and Y directions.
        public float2 normalizers;
        public float2 min, max;

        [WriteOnly]
        public NativeArray<float3> linePositions;

        [NativeDisableContainerSafetyRestriction]
        public NativeReference<float> sum, minT, maxT;

        public void Execute(int startIndex, int count)
        {
            float sumOfBatch = 0f;
            float minOfBatch = 100000f;
            float maxOfBatch = -100f;

            int stopIndex = startIndex + count;
            for (int index = startIndex; index < stopIndex; index++)
            {
                /* 'cursor' points to the oldest value in the buffer.
                 * That value goes first. IE the buffer is offset by 'cursor',
                 * with wraparound. We can undo that easily. */
                int readIndex = index - cursor;
                readIndex = math.select(
                    readIndex,
                    readIndex + times.Length,
                    readIndex < 0
                );
                float time = times[readIndex];
                sumOfBatch += time;
                minOfBatch = math.min(minOfBatch, time);
                maxOfBatch = math.max(maxOfBatch, time);

                float2 norm = normalizers * new float2(index, time);
                float2 positionXY = math.lerp(min, max, norm);

                linePositions[index] = new float3(positionXY, 5f);
            }

            // Atomic float addition.
            ref float sharedValue = ref sum.AsRef();
            float initial, computed;
            do
            {
                initial = sharedValue;
                computed = initial + sumOfBatch;
            }
            while (initial != Interlocked.CompareExchange(ref sharedValue, computed, initial));

            // Atomic float max
            sharedValue = ref maxT.AsRef();
            do
            {
                initial = sharedValue;
                // Abort if the shared value is already larger than the local max.
                if (initial > maxOfBatch) break;
                computed = math.max(initial, maxOfBatch);
            }
            while (initial != Interlocked.CompareExchange(ref sharedValue, computed, initial));

            // Atomic float min
            sharedValue = ref minT.AsRef();
            do
            {
                initial = sharedValue;
                // Abort if the shared value is already smaller than the local min.
                if (initial < minOfBatch) break;
                computed = math.min(initial, minOfBatch);
            }
            while (initial != Interlocked.CompareExchange(ref sharedValue, computed, initial));
        }
    }

    private static Rect window = new Rect(50f, 50f, 250f, 300f);
    private void OnGUI()
    {
        window = GUILayout.Window(GetInstanceID(), window, DrawWindow, "Diagnostics");
        // At most the window may be 50% off-screen.
        float halfWidth = 0.5f * window.width;
        float halfHeight = 0.5f * window.height;
        window.x = Mathf.Clamp(window.x, -halfWidth, Screen.width - halfWidth);
        window.y = Mathf.Clamp(window.y, -halfHeight, Screen.height - halfHeight);
    }

    private static void DrawWindow(int id)
    {
        GUILayout.BeginVertical();
        const string text = "Top of window is: {0} ms\nMean: {1} ms\nMin: {2} ms\nMax: {3} ms";
        GUILayout.Label(string.Format(text, largestTime, MeanOfSamples, MinOfSamples, MaxOfSamples));

        GUILayout.Label("Width Scale");
        widthScale = GUILayout.HorizontalSlider(widthScale, 0f, 1f);
        GUILayout.Label("Height Scale");
        heightScale = GUILayout.HorizontalSlider(heightScale, 0f, 1f);

        /*GUILayout.Label("Top of window is: " + largestTime);
        GUILayout.Label("Mean execution time: " + MeanOfSamples);*/
        GUILayout.EndHorizontal();
        GUI.DragWindow();
    }
}