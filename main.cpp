#include <portaudio.h>
#include <fftw3.h>
#include <iostream>
#include <thread>
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <set>
#include <termios.h>            //termios, TCSANOW, ECHO, ICANON
#include <unistd.h>     //STDIN_FILENO
#include <atomic>
#include <mutex>
#include <gnuplot-iostream.h>
#include <numeric>
#include <zlib.h>

Gnuplot plot("gnuplot 2>/dev/null");
// SOURCE: https://stackoverflow.com/a/1798833
void setunbuf() {
  int c;
  static struct termios oldt, newt;

  /*tcgetattr gets the parameters of the current terminal
  STDIN_FILENO will tell tcgetattr that it should write the settings
  of stdin to oldt*/
  tcgetattr( STDIN_FILENO, &oldt);
  /*now the settings will be copied*/
  newt = oldt;

  /*ICANON normally takes care that one line at a time will be processed
  that means it will return if it sees a "\n" or an EOF or an EOL*/
  newt.c_lflag &= ~(ICANON);

  /*Those new settings will be set to STDIN
  TCSANOW tells tcsetattr to change attributes immediately. */
  tcsetattr( STDIN_FILENO, TCSANOW, &newt);
}

using namespace std::string_literals;
using namespace std::chrono_literals;

constexpr int sample_rate = 44100;
constexpr int max_buffer_size = 4096;
int buffer_size; //sample_rate/64;
double sym_len;
constexpr int n_octaves = 2;
struct freq_params {
  double shift;
  std::vector<double> scale = {
    440 * shift,
    523.25 * shift,
    659.25 * shift,
    783.99 * shift
  };

  double carrier = scale[0] * (1 << n_octaves);
  double sync_begin = scale[1] * (1 << n_octaves);
  double sync_mid = scale[2] * (1 << n_octaves);
  double sync_end = scale[3] * (1 << n_octaves);

  freq_params(double shift_) : shift{shift_} {}
};

class spinlock {
  std::atomic<bool> b{false};

public:
  void lock() {
    while (b.exchange(true));
  }

  void unlock() {
    b = false;
  }
};

struct input {
  freq_params freqs;

  input(double shift) : freqs{shift} {}

  double get_mag(fftw_complex* buf, size_t buf_len, size_t freq) {
    size_t idx = buf_len * freq / sample_rate;
    double ret = 0.;
    for (int nudge = -1; nudge <= 1; ++nudge) {
      auto& elem = buf[idx + nudge];
      ret += (elem[0] * elem[0] + elem[1] * elem[1]) / buf_len;
    }
    return ret;
  }

  std::vector<uint64_t> input_datapoints;

  std::vector<uint64_t> sym_buf;

  uint64_t sym_len;
  int64_t pos_counter;
  uint64_t frame_start;
  enum { SYNC_BEGIN, SYNC_MID, SYNC_END, SYNC_FIN } sync_state = SYNC_FIN;

  spinlock hist_spinlock;
  std::vector<std::array<fftw_complex, max_buffer_size>> hist;
//  std::vector<double> carrier_hist;
//  std::vector<double> sync_begin_hist;
  std::vector<double> quiet_hist;
  size_t quiet_window = 100;
  double encounter_threshold_mov_av = 0.;
  double mov_av_decay = 0.99;
  std::atomic<double> encounter_threshold;

  int64_t remaining_quiet = quiet_window;

  std::array<double, max_buffer_size> dft_in;
  std::array<fftw_complex, max_buffer_size> dft_out;

  fftw_plan plan = fftw_plan_dft_r2c_1d(buffer_size, dft_in.data(), dft_out.data(), FFTW_DESTROY_INPUT);

  void update_quiet(double carrier_mag) {
    encounter_threshold_mov_av *= mov_av_decay;
    encounter_threshold_mov_av += (carrier_mag * (1-mov_av_decay));
    std::unique_lock lock{hist_spinlock};
    quiet_hist.push_back(carrier_mag);
    if (quiet_hist.size() > quiet_window)
      quiet_hist.erase(quiet_hist.begin());
  }

  int handle_input(void const* input, void* output,
                   unsigned long frameCount,
                   PaStreamCallbackTimeInfo const* timeInfo,
                   ::PaStreamCallbackFlags statusFlags,
                   void* userData) {
    pos_counter += 1;

    auto* input_buf = static_cast<float const*>(input);

    std::copy(input_buf, input_buf + frameCount, dft_in.begin());

    ::fftw_execute(plan);

    if (sym_len == 0) {
      input_datapoints.clear();
    }

    if (sym_len && pos_counter && pos_counter % sym_len == 0) {
      if (input_datapoints.size() != 0) {
        std::multiset<uint64_t> freqs(input_datapoints.begin(), input_datapoints.end());
        auto max = std::max_element(freqs.begin(), freqs.end(), [&](auto a, auto b) { return freqs.count(a) < freqs.count(b); });
        input_datapoints.clear();
        sym_buf.push_back(*max);
      }
      else
        sym_buf.push_back(-1);

      if (sym_buf.size() == 2) {
        if (std::find(sym_buf.begin(), sym_buf.end(), -1) == sym_buf.end())
          std::cout << (char)(sym_buf[0] << 4 | sym_buf[1]) << std::flush;;
        sym_buf.clear();
      }
    }

    auto carrier_mag = get_mag(dft_out.data(), frameCount, freqs.carrier);

    if (remaining_quiet) {
      --remaining_quiet;
      update_quiet(carrier_mag);
      return 0;
    }

    decltype(quiet_hist) recent_quiet;
    {
      std::unique_lock lock{hist_spinlock};
      hist.push_back(dft_out);
      recent_quiet = quiet_hist;
    }

    auto initial_mean = std::reduce(recent_quiet.begin(), recent_quiet.end()) / recent_quiet.size();
    auto initial_var = std::accumulate(recent_quiet.begin(), recent_quiet.end(), 0.,
                                       [&](auto acc, auto val) { auto contrib = (val - initial_mean); return acc + (contrib * contrib); });
    auto initial_sd = sqrt(initial_var);

    encounter_threshold = initial_mean + 3*initial_sd;

    if (carrier_mag <= encounter_threshold) {
      update_quiet(carrier_mag);
      return 0;
    }

    auto start_mag = get_mag(dft_out.data(), frameCount, freqs.sync_begin);
    auto mid_mag = get_mag(dft_out.data(), frameCount, freqs.sync_mid);
    auto end_mag = get_mag(dft_out.data(), frameCount, freqs.sync_end);

    auto match_thresh = (carrier_mag + start_mag + mid_mag + end_mag) / 12;

    auto sync_max = std::max({start_mag, mid_mag, end_mag});

//    for (auto i : {})

    if (sync_max > match_thresh) {
      if (start_mag >= sync_max) {
        if (sync_state != SYNC_FIN) {
          if (sync_state != SYNC_MID)
            sync_state = SYNC_FIN;
          return 0;
        }
        sync_state = SYNC_MID;
        input_datapoints.clear();
        return 0;
      }
      else if (mid_mag >= sync_max) {
        if (sync_state != SYNC_MID) {
          if (sync_state != SYNC_END)
            sync_state = SYNC_FIN;
          return 0;
        }
        sym_len = pos_counter;
        sync_state = SYNC_END;
        return 0;
      }
      else {
        if (sync_state != SYNC_END) {
          if (sync_state != SYNC_FIN)
            sync_state = SYNC_FIN;
          return 0;
        }
        sync_state = SYNC_FIN;
        sym_len = (pos_counter - sym_len) / 3;
//        std::cerr << "Receiving message with frame len " << sym_len << std::endl;
        pos_counter = (-sym_len);
        sym_buf.clear();
        input_datapoints.clear();
        return 0;
      }
    }

    uint64_t sym = 0;

    for (int octave = 0; octave < n_octaves; ++octave) {
      size_t max_freq;
      double max_val = -INFINITY;

      for (size_t i = 0; i < freqs.scale.size(); ++i) {
        if (auto x = get_mag(dft_out.data(), frameCount, freqs.scale[i] * (1<<octave)); x > max_val) {
          max_freq = i;
          max_val = x;
        }
      }

      sym <<= 2;
      sym |= max_freq;
    }

    input_datapoints.push_back(sym);

    return 0;
  }
};

struct portaudio_instance {
  portaudio_instance() {
    freopen("/dev/null","w",stderr);
    if (int err = ::Pa_Initialize())
      throw std::runtime_error{"Failed to load PortAudio: "s + Pa_GetErrorText(err)};
    freopen("/dev/tty","w",stderr);
  }
  ~portaudio_instance() {
    if (int err = ::Pa_Terminate())
       throw std::runtime_error{"Failed to unload PortAudio: "s + Pa_GetErrorText(err)};
  }
};

struct output {
  freq_params freqs;

  output(double shift) : freqs{shift} {}

  spinlock msgs_spinlock;
  std::queue<std::string> msgs;
  std::queue<uint8_t> data_out;
  std::array<double, n_octaves + 1> curr_freqs;
  bool has_freqs = false;

  uint64_t sym_pos;
  bool is_starting = false;
  float last = 0;
  uint64_t frame_pos;

  static constexpr double wave_fudge = 5e-1; // 5e-1;

  float sin_from_pos(double freq) {
    return sin(sym_pos * freq * 2 * M_PI / sample_rate);
  }

  float sin_from_pos(double freq, double pos) {
    return sin(pos * freq * 2 * M_PI / sample_rate);
  }

  float start_from_pos() {
    float ret = sin_from_pos(freqs.carrier);
    if (sym_pos < sym_len)
      ret += sin_from_pos(freqs.sync_begin);
    else if (sym_pos < 4*sym_len)
      ret += sin_from_pos(freqs.sync_mid);
    else
      ret += sin_from_pos(freqs.sync_end);

    ++sym_pos;
    return ret / 2;
  }

  float get_from_pos() {
    ++frame_pos;
    if (!is_starting && data_out.size() == 0) {
      if (++sym_pos <= sym_len * 2)
        return 0;
      size_t msgs_size;
      {
        std::unique_lock<spinlock> lock{msgs_spinlock};
        msgs_size = msgs.size();
      }
      if (msgs_size > 0) {
        sym_pos = 0;
        is_starting = true;
      }
      else
        return 0;
    }

    if (is_starting) {
      auto start_val = start_from_pos();
      if (sym_pos <= 5 * sym_len || fabsf(last) > wave_fudge)
        return start_val;
      else {
        is_starting = false;
        std::string msg;
        {
          std::unique_lock<spinlock> lock{msgs_spinlock};
          msg = std::move(msgs.front());
          msgs.pop();
        }
        for (auto& i : msg) {
          data_out.push(i >> 4);
          data_out.push(i & 0xf);
        }
        sym_pos = 0;
        frame_pos = 0;
      }
    }

    if (!has_freqs) {
      auto dat_acc = data_out.front();
      for (size_t i = 0; i < n_octaves; ++i) {
        curr_freqs[i] = freqs.scale[dat_acc & 3] *  (1<<(n_octaves-i - 1));
        dat_acc >>= 2;
      }
      curr_freqs[n_octaves] = freqs.carrier;
      has_freqs = true;
    }

    if (sym_pos < sym_len || fabsf(last) > wave_fudge) {
      float ret = 0;
      for (auto freq: curr_freqs)
        ret += sin_from_pos(freq, frame_pos);
      ++sym_pos;
      return ret / curr_freqs.size();
    }
    else {
      sym_pos -= sym_len;
      has_freqs = false;
      data_out.pop();
      if (data_out.size() == 0)
        return 0;
      return get_from_pos();
    }
  }

  int handle_output(void const* input, void* output,
                    unsigned long frameCount,
                    PaStreamCallbackTimeInfo const* timeInfo,
                    ::PaStreamCallbackFlags statusFlags,
                    void* userData) {
    static uint64_t sym_pos = 0;
  //  constexpr double sym_len = sample_rate / data_rate;

    auto buf = reinterpret_cast<float*>(output);

    for (size_t i = 0; i < frameCount; ++i) {
      buf[i] = get_from_pos();
      last = buf[i];
    }

    return 0;
  }
};

portaudio_instance _pa;

int forward_input(void const* input, void* output,
                   unsigned long frameCount,
                   PaStreamCallbackTimeInfo const* timeInfo,
                   ::PaStreamCallbackFlags statusFlags,
                   void* userData) {
  return reinterpret_cast<struct input*>(userData)->handle_input(input, output, frameCount, timeInfo, statusFlags, userData);
}

int forward_output(void const* input, void* output,
                   unsigned long frameCount,
                   PaStreamCallbackTimeInfo const* timeInfo,
                   ::PaStreamCallbackFlags statusFlags,
                   void* userData) {
  return reinterpret_cast<struct output*>(userData)->handle_output(input, output, frameCount, timeInfo, statusFlags, userData);
}


int main(int argc, char** argv) {
  std::string line;

  PaStream* out, * in;

  auto sym_frames = argc >= 2 ? std::stof(argv[1]) : 4;

  sym_len = buffer_size = argc >= 4 ? std::stoi(argv[3]) : 256; //1024;
  output out_s{argc >= 3 ? std::stod(argv[2]) : 3};
  input in_s{argc >= 3 ? std::stod(argv[2]) : 3};

  if (int err = ::Pa_OpenDefaultStream(&in, 1, 0, paFloat32, sample_rate, buffer_size, &forward_input, &in_s))
    throw std::runtime_error{"Failed to open input: "s + Pa_GetErrorText(err)};
  if (int err = ::Pa_OpenDefaultStream(&out, 0, 1, paFloat32, sample_rate, buffer_size, &forward_output, &out_s))
    throw std::runtime_error{"Failed to open output: "s + Pa_GetErrorText(err)};
  if (int err = Pa_StartStream(in))
    throw std::runtime_error{"Failed to start input: "s + Pa_GetErrorText(err)};

  auto get_mag = [&](auto freq, auto elem) {
    return in_s.get_mag(elem.data(), buffer_size, freq);
  };

  std::thread scale_monitor{[&] {
      Gnuplot scale_plot("gnuplot 2>/dev/null");

      scale_plot << "set spiderplot; set style spiderplot lw 2.0\n"; //fillstyle solid 0.2 border

      auto loop_start = std::chrono::high_resolution_clock::now();
      for (uint64_t loop_n = 0; true; ++loop_n) {
        std::this_thread::sleep_until(loop_start + (1s/24));
        loop_start = std::chrono::high_resolution_clock::now();
        decltype(in_s.hist)::value_type latest;
        {
          std::unique_lock lock{in_s.hist_spinlock};
          if (!in_s.hist.size())
            continue;
          latest = in_s.hist.back();
        }

        std::vector<float> upper, lower;
        for (auto i : in_s.freqs.scale) {
          lower.push_back(get_mag(i, latest));
          upper.push_back(get_mag(i * 2, latest));
        }
        if (loop_n % 24*3) {
          auto max = std::max(*std::max_element(upper.begin(), upper.end()), *std::max_element(lower.begin(), lower.end()));
          scale_plot << "set for [i=1:100] paxis i linewidth 1 range [0:" << max << "]\n";
        }
        scale_plot << "$DATA << EOF\n";
        std::string dat = "1 2 3 4\n";
        for (auto i : upper)
          dat += std::to_string(i) + ' ';
        dat.back() = '\n';
        for (auto i : lower)
          dat += std::to_string(i) + ' ';
        dat.back() = '\n';
        dat += "EOF\n";
        scale_plot << dat
                   << "plot for [i=1:4] $DATA using i title columnhead\n"
                   << std::flush;
      }
  }};

  std::thread hist_monitor{[&] {
      size_t carrier_plot_width = sym_frames * 100;
      Gnuplot hist_plot("gnuplot 2>/dev/null");
      hist_plot << "set logscale y\n"
                   "set xrange [0:" << carrier_plot_width - 1 << "]\n";

      while (true) {
        std::this_thread::sleep_for(1s/60);
        decltype(in_s.hist) hist;
        {
          std::unique_lock lock{in_s.hist_spinlock};
          hist.insert(hist.end(), in_s.hist.begin() + hist.size(), in_s.hist.end());
          hist = in_s.hist;
          if (hist.size() > carrier_plot_width)
            hist.erase(hist.end() - carrier_plot_width, hist.end());
        }

        auto offset = (hist.size() < carrier_plot_width) ? 0 : hist.size() - carrier_plot_width;
        std::vector<double> carrier_plot, sync_begin_plot, sync_mid_plot, sync_end_plot, data_plot;
        for (size_t i = offset; i < hist.size(); ++i) {
          carrier_plot.push_back(get_mag(in_s.freqs.carrier, hist[i]));
          sync_begin_plot.push_back(get_mag(in_s.freqs.sync_begin, hist[i]));
          sync_mid_plot.push_back(get_mag(in_s.freqs.sync_mid, hist[i]));
          sync_end_plot.push_back(get_mag(in_s.freqs.sync_end, hist[i]));
          data_plot.push_back(0.);
          for (auto freq : in_s.freqs.scale) {
            data_plot.back() += get_mag(freq, hist[i]) + get_mag(freq * 2, hist[i]);
          }
        }
        hist_plot << "plot '-' with line title 'carrier', "
                "'-' with line title 'sync begin', "
                "'-' with line title 'sync mid', "
                "'-' with line title 'sync end', "
                "'-' with line title 'data' lc rgb'red', "
             << in_s.encounter_threshold << " title 'threshold'\n";
        hist_plot.send1d(carrier_plot);
        hist_plot.send1d(sync_begin_plot);
        hist_plot.send1d(sync_mid_plot);
        hist_plot.send1d(sync_end_plot);
        hist_plot.send1d(data_plot);
      }
  }};

  if (argc != 1) {
    sym_len *= sym_frames;
    if (int err = Pa_StartStream(out))
      throw std::runtime_error{"Failed to start output: "s + Pa_GetErrorText(err)};
    while (std::getline(std::cin, line)) {
      std::unique_lock<spinlock> lock{out_s.msgs_spinlock};
      out_s.msgs.push(line + "\n");
    }
  }

  hist_monitor.join();
  scale_monitor.join();
}
