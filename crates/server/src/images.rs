//! Images in requests: the embedded image of a content part (a `data:` URL,
//! or base64 alone as llama-server takes it) decoded to RGB pixels for the
//! model ([`ares_api::Image`]).
//!
//! JPEG, PNG, GIF (its first frame), WebP and BMP are read. Like llama.cpp's
//! `stb_image`, the decoder drops an alpha channel and applies no EXIF
//! orientation or colour profile; 16-bit channels are rounded to 8. Fetching an
//! image from a remote URL is not built: such a part is refused with a 400 that
//! says to send the image itself.
//!
//! Limits: the request body's own (64 MiB), and an image of at most
//! [`MAX_SIDE`] pixels a side and [`MAX_PIXELS`] pixels, read from its header
//! before anything is decoded. How many rows an image becomes is the model's
//! business (its projector resizes it, `--image-max-tokens`).

use std::sync::Arc;

use ares_api::Image;

use crate::error::ApiError;

/// The most pixels on a side of an image a request may send.
pub const MAX_SIDE: u32 = 16384;
/// The most pixels of an image a request may send: a 64-megapixel photo.
pub const MAX_PIXELS: u64 = 64 << 20;

/// The pixels of an image a part of a request names, decoded.
pub struct Decoded {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Decoded {
    /// The image at byte `at` of a message's text.
    pub fn at(self, at: usize) -> Image {
        Image { at, width: self.width, height: self.height, pixels: Arc::new(self.pixels) }
    }
}

/// The image of a URL of a content part (`image_url.url`, `input_image`'s
/// `image_url`): a `data:image/...;base64,` URL, or base64 alone. `at` names
/// the part in an error.
pub fn from_url(at: &str, url: &str) -> Result<Decoded, ApiError> {
    let refuse = |message: String| ApiError::invalid_param(at, format!("{at}: {message}"));
    if url.starts_with("http://") || url.starts_with("https://") {
        return Err(refuse(
            "images are not fetched from URLs: send the image itself, as a data URL \
             (data:image/png;base64,...)"
                .into(),
        ));
    }
    let data = match url.strip_prefix("data:") {
        Some(rest) => {
            let Some((head, data)) = rest.split_once(',') else {
                return Err(refuse("a data URL without a ',' before its data".into()));
            };
            if !head.starts_with("image/") {
                return Err(refuse(format!(
                    "a data URL of the type '{}' is not an image",
                    head.split(';').next().unwrap_or(head)
                )));
            }
            if !head.ends_with(";base64") {
                return Err(refuse("a data URL of an image must be base64 (;base64,)".into()));
            }
            data
        }
        None if url.contains("://") => {
            return Err(refuse(format!(
                "'{}' URLs are not read: send the image as a data URL",
                url.split("://").next().unwrap_or(url)
            )));
        }
        None => url,
    };
    from_base64(at, data)
}

/// The image of base64 data (Anthropic's `source.data`).
pub fn from_base64(at: &str, data: &str) -> Result<Decoded, ApiError> {
    let refuse = |message: String| ApiError::invalid_param(at, format!("{at}: {message}"));
    let bytes = crate::base64::decode_lenient(data)
        .ok_or_else(|| refuse("the image's base64 does not decode".into()))?;
    decode(&bytes).map_err(refuse)
}

/// Decodes the bytes of an image file to RGB pixels, three bytes a pixel.
pub fn decode(bytes: &[u8]) -> Result<Decoded, String> {
    use image::ImageReader;
    let reader = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("the image cannot be read: {e}"))?;
    if reader.format().is_none() {
        return Err("the image is not of a format that is read (JPEG, PNG, GIF, WebP, BMP)".into());
    }
    let (width, height) =
        reader.into_dimensions().map_err(|e| format!("the image cannot be read: {e}"))?;
    if width == 0 || height == 0 {
        return Err("the image has no pixels".into());
    }
    if width > MAX_SIDE || height > MAX_SIDE || u64::from(width) * u64::from(height) > MAX_PIXELS {
        return Err(format!(
            "the image is {width} x {height} pixels; at most {MAX_SIDE} a side and {} \
             megapixels are read",
            MAX_PIXELS >> 20
        ));
    }
    let mut reader = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| format!("the image cannot be read: {e}"))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(MAX_SIDE);
    limits.max_image_height = Some(MAX_SIDE);
    limits.max_alloc = Some(MAX_PIXELS * 16);
    reader.limits(limits);
    let decoded = reader.decode().map_err(|e| format!("the image does not decode: {e}"))?;
    let rgb = decoded.into_rgb8();
    Ok(Decoded { width: rgb.width(), height: rgb.height(), pixels: rgb.into_raw() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A PNG of 3 x 2 pixels, red, green, blue / white, black, gray, with an
    /// alpha channel that the decoder drops.
    fn png() -> Vec<u8> {
        let pixels: Vec<u8> = [
            [255, 0, 0, 255],
            [0, 255, 0, 128],
            [0, 0, 255, 0],
            [255, 255, 255, 255],
            [0, 0, 0, 255],
            [128, 128, 128, 255],
        ]
        .concat();
        let mut bytes = Vec::new();
        image::RgbaImage::from_raw(3, 2, pixels)
            .unwrap()
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        bytes
    }

    fn encode64(bytes: &[u8]) -> String {
        let mut out = String::new();
        crate::base64::encode(bytes, &mut out);
        out
    }

    #[test]
    fn a_data_url_of_a_png_becomes_its_rgb_pixels() {
        let url = format!("data:image/png;base64,{}", encode64(&png()));
        let image = from_url("messages[0].content[1].image_url", &url).ok().unwrap();
        assert_eq!((image.width, image.height), (3, 2));
        assert_eq!(&image.pixels[..9], &[255, 0, 0, 0, 255, 0, 0, 0, 255]);
        assert_eq!(&image.pixels[9..], &[255, 255, 255, 0, 0, 0, 128, 128, 128]);
        // Base64 alone, as llama-server takes it.
        assert!(from_url("x", &encode64(&png())).is_ok());
    }

    #[test]
    fn remote_urls_other_types_and_damaged_images_are_refused_by_name() {
        let said = |url: &str| match from_url("messages[0].content[1].image_url", url) {
            Ok(_) => panic!("{url} was taken"),
            Err(error) => error.message,
        };
        assert!(said("https://example.com/cat.png").contains("data URL"));
        assert!(said("file:///etc/passwd").contains("'file' URLs"));
        assert!(said("data:text/plain;base64,aGk=").contains("not an image"));
        assert!(said("data:image/png,rawbytes").contains("base64"));
        assert!(said("data:image/png;base64,aGVsbG8=").contains("not of a format"));
        let mut cut = png();
        cut.truncate(cut.len() - 20);
        assert!(said(&format!("data:image/png;base64,{}", encode64(&cut))).contains("decode"));
    }
}
