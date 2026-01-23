import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ThreeJS.Bridge",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ThreeJSRenderNode") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // --- Node Data & UI Elements ---
                const nodeId = this.id;
                let currentRenderConfig = {}; // 存储最新的渲染配置，用于预览

                // --- DOM 结构 ---
                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.gap = "5px";
                
                const iframe = document.createElement("iframe");
                // 修正：使用更稳健的 URL 获取方式
                iframe.src = new URL("./three_renderer.html", import.meta.url).href;
                iframe.style.width = "100%";
                iframe.style.height = "100%"; 
                iframe.style.border = "1px solid #333";
                
                const btnPanel = document.createElement("div");
                const previewBtn = document.createElement("button");
                previewBtn.innerText = "▶ 实时预览 / 停止";
                previewBtn.style.width = "100%";
                previewBtn.onclick = () => {
                    if (Object.keys(currentRenderConfig).length > 0) {
                        iframe.contentWindow.postMessage({ 
                            type: "START_PREVIEW",
                            config: currentRenderConfig 
                        }, "*");
                    } else {
                        alert("请先运行一次 ComfyUI 提示，以加载渲染配置到预览器。");
                    }
                };
                
                btnPanel.appendChild(previewBtn);
                container.appendChild(iframe);
                container.appendChild(btnPanel);

                // --- ComfyUI Widget Definition ---
                this.addDOMWidget("HTML_PREVIEW", "three_preview", container, {
                    getValue() { return ""; },
                    setValue(v) {},
                    computeSize(node) {
                        if (currentRenderConfig.width && currentRenderConfig.height) {
                            const aspectRatio = currentRenderConfig.width / currentRenderConfig.height;
                            // 计算一个理想的高度，但允许 ComfyUI 在一定程度上自由缩放
                            // 这里返回尺寸主要是为了节点初始化时不要太扁或太长
                            const newWidgetWidth = 512; 
                            const newWidgetHeight = Math.round(newWidgetWidth / aspectRatio) + 30; 
                            
                            // 移除 node.setSize 强制设置，因为这会阻止用户手动拖拽
                            // node.setSize([newWidgetWidth, newWidgetHeight]); 
                            
                            return [newWidgetWidth, newWidgetHeight];
                        }
                        return [512, 450];
                    }
                });

                // --- 通信监听 ---
                const handleRenderRequest = (event) => {
                    const { detail } = event;
                    if (detail.node_id && detail.node_id !== nodeId) return;

                    console.log("[ThreeJS Bridge] Received render request, forwarding to iframe...");
                    currentRenderConfig = detail; 
                    
                    // --- 修复点：删除了错误的 app.graph.set.-1; ---
                    app.graph.setDirtyCanvas(true, true);

                    if (iframe.contentWindow) {
                        iframe.contentWindow.postMessage({
                            type: "RENDER_BATCH",
                            config: detail
                        }, "*");
                    } else {
                        console.error("[ThreeJS Bridge] Iframe not ready!");
                    }
                };
                
                api.addEventListener("threejs_render_request", handleRenderRequest);

                // 监听 iframe 发回的结果
                window.addEventListener("message", async (event) => {
                    if (event.source !== iframe.contentWindow) return;
                    const data = event.data;
                    
                    if (data.type === "RENDER_RESULT") {
                        console.log(`[ThreeJS Bridge] Uploading ${data.images.length} frames to backend...`);
                        try {
                            await fetch("/threejs/render_result", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({
                                    request_id: data.request_id,
                                    images: data.images
                                })
                            });
                        } catch (err) {
                            console.error("[ThreeJS Bridge] Upload failed", err);
                        }
                    }
                });

                // --- 节点生命周期钩子 ---
                const originalOnExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    if (originalOnExecuted) originalOnExecuted.apply(this, arguments);
                    app.graph.setDirtyCanvas(true, true);
                };

                return r;
            };
        }
    }
});