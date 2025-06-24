export async function cropFaceToDataURL(imageUrl, faceLocation) {
  const [top, right, bottom, left] = faceLocation;

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous"; // important for local dev or remote APIs
    img.src = imageUrl;

    img.onload = () => {
      const width = right - left;
      const height = bottom - top;

      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;

      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, left, top, width, height, 0, 0, width, height);

      resolve(canvas.toDataURL()); // base64 string
    };

    img.onerror = reject;
  });
}
