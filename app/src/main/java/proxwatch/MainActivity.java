package proxwatch;

import android.Manifest;
import android.bluetooth.BluetoothClass;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.le.ScanResult;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.os.Handler;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebView;
import android.widget.Button;
import android.widget.TextView;



import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.HashMap;
import java.util.Locale;
import java.util.TimeZone;

public class MainActivity extends AppCompatActivity {

    private Handler mHandler;

    int LIST_THRESH = -60; //minimal signal strength to show up in the list
    //hashMap that stores bt device info index by address
    HashMap<String, ScanResult> results = new HashMap<String, ScanResult>();
    boolean isVisible = false; //keep trck of whether this activity is in the foreground

    String generateChartString(int type) { //format the hourly exposure data into a format that can be sent to quickchart.. Type  1=hour, type 2=day, type 3 = week

        //chart fomratting data
        String BASE_REQUEST_HOUR = "https://quickchart.io/chart?c={type:%27line%27,%20options:%20{legend:%20{display:%20false}},data:{labels:[%2712%20AM%27,%271%20AM%27,%272%20AM%27,%273%20AM%27,%274%20AM%27,%275%20AM%27,%276%20AM%27,%277%20AM%27,%278%20AM%27,%279%20AM%27,%2710%20AM%27,%2711%20AM%27,%2712%20PM%27,%20%271%20PM%27,%272%20PM%27,%273%20PM%27,%274%20PM%27,%275%20PM%27,%276%20PM%27,%277%20PM%27,%278%20PM%27,%279%20PM%27,%2710%20PM%27,%2711%20PM%27],%20datasets:[{label:%27%27,%20data:%20[#CHARTDATA#],%20fill:false,borderColor:%27maroon%27}]}}";
        String BASE_REQUEST_MINUTE = "https://quickchart.io/chart?c={type:%27line%27,%20options:%20{legend:%20{display:%20false}},data:{labels:[#LABELDATA#],%20datasets:[{label:%27%27,%20data:%20[#CHARTDATA#],%20fill:false,borderColor:%27maroon%27}]}}";
        String BASE_REQUEST_DAILY = "https://quickchart.io/chart?c={type:'line', options: {legend: {display: false}},data:{labels:['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'], datasets:[{label:'', data: [#CHARTDATA#], fill:false,borderColor:'maroon'}]}}";

        //Strings for obtained data
        String hourlyData = "";
        String dailyData = "";
        String minuteData = "";

        //formats and keys
        SimpleDateFormat todayFormat = new SimpleDateFormat("dd-MMM-yyyy");
        String todayKey = todayFormat.format(Calendar.getInstance(TimeZone.getDefault(), Locale.getDefault()).getTime());
        SimpleDateFormat weekdayFormat = new SimpleDateFormat("-W-MMM-yyyy");
        String weekdayKey = weekdayFormat.format(Calendar.getInstance(TimeZone.getDefault(), Locale.getDefault()).getTime());
        SimpleDateFormat minuteFormat = new SimpleDateFormat("-H-dd-MMM-yyyy");
        String minuteKey = minuteFormat.format(Calendar.getInstance(TimeZone.getDefault(), Locale.getDefault()).getTime());
        SimpleDateFormat hourFormat = new SimpleDateFormat("H");
        final SharedPreferences prefs = getSharedPreferences("com", MODE_PRIVATE);

        int thisHour = Integer.parseInt(hourFormat.format(Calendar.getInstance(TimeZone.getDefault(), Locale.getDefault()).getTime())); //todo: clearer way of getting current hour

        // /go through each hour of the day and append teh number of contacts to hourlydata.
        // We want to plot points for all the hours that have actually happened. Some might be missing data (if the app wasn't running), so show these as 0
        for (int time = 0; time <= thisHour; time++) {
            int contactNum = prefs.getInt(time + "-" + todayKey, -1);
            if (contactNum > -1) { //we actually have data for this slot
                hourlyData = hourlyData + contactNum + ",";
            } else {
                hourlyData = hourlyData + "0,";
            }
        }
        //now do the same thing for minute
        for (int min = 0; min <= 59; min++) {
            int contactNum = prefs.getInt("min-" + min + minuteKey, -1);


            if (contactNum > -1) { //we actually have data for this slot
                minuteData = minuteData + contactNum + ",";
            } else {
                minuteData = minuteData + "0,";
            }
        }

        //and for the day of the week
        for (int day = 1; day <= 7; day++) {
            int contactNum = prefs.getInt("week-" + day + weekdayKey, -1);
            if (contactNum > -1) { //we actually have data for this slot
                dailyData = dailyData + contactNum + ",";
            } else {
                dailyData = dailyData + "0,";
            }
        }

        //return the URL for the request chart. 1=current hour, 2=current day, 3=current week
        if (type == 1) {
            String minLabel = "";
            for (int i = 0; i < 60; i++) { //create labels from 1 to 60 minutes
                minLabel = minLabel + i + ",";
            }
            BASE_REQUEST_MINUTE = BASE_REQUEST_MINUTE.replace("#LABELDATA#", minLabel).replace("#CHARTDATA#", minuteData);
            return BASE_REQUEST_MINUTE;
        } else if (type == 2) {
            BASE_REQUEST_HOUR = BASE_REQUEST_HOUR.replace("#CHARTDATA#", hourlyData); //plug the data into the URL
            return BASE_REQUEST_HOUR;
        } else {
            BASE_REQUEST_DAILY = BASE_REQUEST_DAILY.replace("#CHARTDATA#", dailyData); //plug the data into the URL
            return BASE_REQUEST_DAILY;
        }
    }

    boolean isWearable(BluetoothDevice bd) {
        int btClass = bd.getBluetoothClass().getDeviceClass();
        int majorClass = bd.getBluetoothClass().getMajorDeviceClass();
        return (true || majorClass == BluetoothClass.Device.Major.UNCATEGORIZED || majorClass == BluetoothClass.Device.Major.PHONE || majorClass == BluetoothClass.Device.Major.WEARABLE || majorClass == BluetoothClass.Device.Major.HEALTH || btClass == BluetoothClass.Device.AUDIO_VIDEO_HEADPHONES || btClass == BluetoothClass.Device.AUDIO_VIDEO_WEARABLE_HEADSET || btClass == BluetoothClass.Device.AUDIO_VIDEO_HANDSFREE);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Bundle extras = getIntent().getExtras();
        if (getIntent().getBooleanExtra("btReset", false)) { //do a silent start just to reinitialize Bluetooth stuff
            Intent intent = new Intent(MainActivity.this, MyForeGroundService.class);
            intent.setAction(MyForeGroundService.ACTION_START_FOREGROUND_SERVICE);
            startService(intent);
            finish();
        } else {

            setContentView(R.layout.activity_main);
            final SharedPreferences prefs = getSharedPreferences("com", MODE_PRIVATE);
            final SharedPreferences.Editor editor = prefs.edit();


            //set up the exposure chart using quickchart.io to make the chart and Webview to display it
            final WebView chartView = (WebView) findViewById(R.id.chartView);
            chartView.setInitialScale(30);
            chartView.setBackgroundColor(Color.WHITE);
            chartView.getSettings().setLoadWithOverviewMode(true); //set scaling to automatically fit the image returned by server
            chartView.setScrollBarStyle(WebView.SCROLLBARS_OUTSIDE_OVERLAY);
            chartView.setScrollbarFadingEnabled(false);
            chartView.getSettings().setUseWideViewPort(true);



            //update the screen with list of detected devices
            final TextView status = (TextView) findViewById(R.id.scanResults);
            final TextView contactsToday = (TextView) findViewById(R.id.contactsList);
            final Handler handler = new Handler();
            final Runnable updateLoop = new Runnable() {
                @Override
                public void run() {
                    // first update the total number of contacts today
                    SimpleDateFormat todayFormat = new SimpleDateFormat("dd-MMM-yyyy");
                    String todayKey = todayFormat.format(Calendar.getInstance(TimeZone.getDefault(), Locale.getDefault()).getTime());
                    contactsToday.setText("My Social Distance Score: " + prefs.getInt(todayKey, 0));
                    if (isVisible) {
                        chartView.loadUrl(generateChartString(prefs.getInt("chartMode", 2))); //update the chart
                    }

                    //show the devices contirbuting--this is not visible by default because the textView that holds it is set to GONE but can be turned pn
                    String dispResult = "";
                    for (String i : scanData.getInstance().getData().keySet()) {
                        ScanResult temp = scanData.getInstance().getData().get(i);
                        if (temp.getRssi() > LIST_THRESH) {
                            dispResult = dispResult + temp.getDevice().getAddress() + " : " + temp.getDevice().getName() + " " + temp.getRssi() + "\n";
                        }
                    }
                    status.setText(dispResult);

                    handler.postDelayed(this, 30000);

                }

            };
// start
            handler.post(updateLoop);

            if (ContextCompat.checkSelfPermission(this,
                    Manifest.permission.ACCESS_COARSE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
//start the bluetooth search service, if we have the required location permission
                Intent intent = new Intent(MainActivity.this, MyForeGroundService.class);
                intent.setAction(MyForeGroundService.ACTION_START_FOREGROUND_SERVICE);
                startService(intent);
            } else { //otherwise this is probably the first run, so open the intro window
                MainActivity.this.startActivity(new Intent(MainActivity.this, PrivacySetup.class));
            }


            //buttons for controlling the time view
            final Button viewHour = (Button) findViewById(R.id.viewHour);
            viewHour.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    editor.putInt("chartMode", 1);
                    chartView.loadUrl(generateChartString(1));
                    editor.apply();
                }
            });


            final Button viewDay = (Button) findViewById(R.id.viewDay);
            viewDay.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    editor.putInt("chartMode", 2);
                    chartView.loadUrl(generateChartString(2));
                    editor.apply();
                }
            });

            final Button viewWeek = (Button) findViewById(R.id.viewWeek);
            viewWeek.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    editor.putInt("chartMode", 3);
                    chartView.loadUrl(generateChartString(3));
                    editor.apply();
                }
            });
        }
    }


    @Override
    protected void onResume() { //start the scan when the application starts
        super.onResume();
        isVisible = true; //the app is visible
    }

    @Override
    protected void onPause() { //start the scan when the application starts
        super.onPause();
        isVisible = false; //the app is no longer visible
    }

    @Override
    protected void onUserLeaveHint() {
        super.onUserLeaveHint();
        finish();
    }

}





