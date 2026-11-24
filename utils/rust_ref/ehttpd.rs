#![forbid(unsafe_code)]

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const RESP_KA: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 13\r\nConnection: keep-alive\r\n\r\nHello, Eon!\r\n";
const RESP_CLOSE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 13\r\nConnection: close\r\n\r\nHello, Eon!\r\n";
const RESP_BYE: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbye\n";

fn parse_req(req: &[u8]) -> (bool, bool) {
    let mut ps = 0usize;
    while ps < req.len() && req[ps] != b' ' {
        ps += 1;
    }
    if ps < req.len() {
        ps += 1;
    }
    let mut pe = ps;
    while pe < req.len() && req[pe] != b' ' && req[pe] != b'\r' && req[pe] != b'\n' {
        pe += 1;
    }
    let is_quit = &req[ps..pe] == b"/quit";
    let has_close = req
        .windows(17)
        .any(|w| (w[0] == b'C' || w[0] == b'c') && w[1] == b'o' && w[2] == b'n' && w[10] == b':' && w[12] == b'c' && w[13] == b'l');
    (is_quit, !has_close && !is_quit)
}

fn handle_conn(
    mut stream: TcpStream,
    echo_mode: bool,
    max_reqs: u64,
    total_reqs: &AtomicU64,
    stop: &AtomicBool,
) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(Duration::from_millis(200)));
    let mut buf = vec![0u8; 2048];
    loop {
        if stop.load(Ordering::Acquire) {
            break;
        }
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                let req_id = total_reqs.fetch_add(1, Ordering::SeqCst) + 1;
                if echo_mode {
                    if stream.write_all(&buf[..n]).is_err() {
                        break;
                    }
                } else {
                    let (is_quit, keep_alive) = parse_req(&buf[..n]);
                    let resp = if is_quit {
                        RESP_BYE
                    } else if keep_alive {
                        RESP_KA
                    } else {
                        RESP_CLOSE
                    };
                    if stream.write_all(resp).is_err() {
                        break;
                    }
                    if is_quit {
                        stop.store(true, Ordering::SeqCst);
                        break;
                    }
                    if !keep_alive {
                        break;
                    }
                }
                if max_reqs > 0 && req_id >= max_reqs {
                    stop.store(true, Ordering::SeqCst);
                    break;
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock || e.kind() == std::io::ErrorKind::TimedOut => {
                if stop.load(Ordering::Acquire) {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut port: u16 = 18081;
    let mut threads: usize = 4;
    let mut max_reqs: u64 = 0;
    let mut echo_mode = false;
    let mut i = 1usize;
    while i < args.len() {
        if args[i] == "--port" && i + 1 < args.len() {
            port = args[i + 1].parse().unwrap_or(port);
            i += 2;
        } else if args[i] == "--threads" && i + 1 < args.len() {
            threads = args[i + 1].parse().unwrap_or(threads).clamp(1, 64);
            i += 2;
        } else if args[i] == "--max-reqs" && i + 1 < args.len() {
            max_reqs = args[i + 1].parse().unwrap_or(0);
            i += 2;
        } else if args[i] == "--echo" {
            echo_mode = true;
            i += 1;
        } else {
            i += 1;
        }
    }

    let listener = TcpListener::bind(("127.0.0.1", port)).expect("bind failed");
    listener.set_nonblocking(true).expect("set_nonblocking failed");
    let total_reqs = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let queue: Arc<Mutex<Vec<TcpStream>>> = Arc::new(Mutex::new(Vec::new()));

    // Spawn worker threads handling accepted connections concurrently (thread-pool)
    // Plus allow up to max(threads * 16, 64) concurrent keep-alive connections cleanly
    let worker_count = (threads * 16).max(64);
    let mut handles = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let q = Arc::clone(&queue);
        let tr = Arc::clone(&total_reqs);
        let st = Arc::clone(&stop);
        handles.push(thread::spawn(move || {
            while !st.load(Ordering::Acquire) {
                let maybe = {
                    let mut g = q.lock().unwrap();
                    g.pop()
                };
                if let Some(stream) = maybe {
                    handle_conn(stream, echo_mode, max_reqs, &tr, &st);
                } else {
                    thread::sleep(Duration::from_micros(100));
                }
            }
        }));
    }

    println!(
        "ehttpd_rs READY port={} threads={} mode={}",
        port,
        threads,
        if echo_mode { "echo" } else { "http" }
    );
    let _ = std::io::stdout().flush();

    while !stop.load(Ordering::Acquire) {
        match listener.accept() {
            Ok((stream, _)) => {
                queue.lock().unwrap().push(stream);
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(Duration::from_micros(50));
            }
            Err(_) => break,
        }
    }

    for h in handles {
        let _ = h.join();
    }
    println!("ehttpd_rs DONE reqs={}", total_reqs.load(Ordering::SeqCst));
}
